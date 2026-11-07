"""Optimal market making with inventory risk - pre-computed HJB lookup."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 280  # Safety margin before 330s kill
_policy_cache = {}


def _solve_regime(sigma, A, k, gamma, Q, steps, T):
    """Solve the discrete HJB backward induction for a given regime."""
    dt = T / steps
    n_q = 2 * Q + 1  # inventory states: -Q, ..., Q
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # Depth grid for searching optimal depths
    # Use a reasonably fine grid; depths are typically in [0.01, 5.0]
    d_min = 0.005
    d_max = 5.0
    n_d = 200
    depth_grid = np.linspace(d_min, d_max, n_d)
    
    # Precompute fill probabilities for each depth
    # prob_fill(d) = 1 - exp(-A * exp(-k*d) * dt)
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # Value function: V[n, q_idx] = max expected reward from step n onward
    # V has shape (steps+1, n_q)
    V = np.zeros((steps + 1, n_q), dtype=np.float64)
    policy = np.zeros((steps, n_q, 2), dtype=np.float64)  # bid_depth, ask_depth
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        for qi in range(n_q):
            q = q_vals[qi]
            
            # Penalty for current inventory
            penalty = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            # Try all bid/ask depth combinations
            # For speed, we'll use a coarser grid first, then refine
            # Actually, let's just use the full grid but vectorize where possible
            
            best_val = -np.inf
            best_bid = depth_grid[0]
            best_ask = depth_grid[0]
            
            # Bid depth options
            if q < Q:
                bid_depths = depth_grid
            else:
                bid_depths = np.array([np.inf])
            
            # Ask depth options
            if q > -Q:
                ask_depths = depth_grid
            else:
                ask_depths = np.array([np.inf])
            
            # For each combination, compute expected value
            # If bid fills: earn bid_depth, inventory becomes q+1
            # If ask fills: earn ask_depth, inventory becomes q-1
            # If neither: no fill
            # If both: earn bid_depth + ask_depth, inventory stays q
            
            for bi, bd in enumerate(bid_depths):
                if np.isinf(bd):
                    p_bid = 0.0
                    bid_earn = 0.0
                else:
                    p_bid = fill_probs[bi]
                    bid_earn = bd
                
                for ai, ad in enumerate(ask_depths):
                    if np.isinf(ad):
                        p_ask = 0.0
                        ask_earn = 0.0
                    else:
                        p_ask = fill_probs[ai]
                        ask_earn = ad
                    
                    # Probabilities
                    p_both = p_bid * p_ask
                    p_bid_only = p_bid * (1 - p_ask)
                    p_ask_only = (1 - p_bid) * p_ask
                    p_none = (1 - p_bid) * (1 - p_ask)
                    
                    # Expected reward from fills
                    if q < Q:
                        bid_next_q = q + 1
                        bid_next_idx = int(bid_next_q + Q)
                    else:
                        bid_next_idx = qi  # no change if at boundary
                    
                    if q > -Q:
                        ask_next_q = q - 1
                        ask_next_idx = int(ask_next_q + Q)
                    else:
                        ask_next_idx = qi  # no change if at boundary
                    
                    # Expected continuation value
                    if np.isinf(bd) or np.isinf(ad):
                        # No fills possible
                        cont_val = V[n + 1, qi]
                        fill_reward = 0.0
                    else:
                        # Both fills: inventory stays q
                        # Bid only: inventory q+1
                        # Ask only: inventory q-1
                        # None: inventory q
                        cont_val = (p_both * V[n + 1, qi] +
                                    p_bid_only * V[n + 1, bid_next_idx] +
                                    p_ask_only * V[n + 1, ask_next_idx] +
                                    p_none * V[n + 1, qi])
                        fill_reward = (p_both * (bid_earn + ask_earn) +
                                       p_bid_only * bid_earn +
                                       p_ask_only * ask_earn)
                    
                    total_val = fill_reward - penalty + cont_val
                    
                    if total_val > best_val:
                        best_val = total_val
                        best_bid = bd
                        best_ask = ad
            
            V[n, qi] = best_val
            policy[n, qi, 0] = best_bid
            policy[n, qi, 1] = best_ask
    
    return policy


def _get_policy(sigma, A, k, gamma, Q, steps, T):
    """Get or compute the policy for a given regime."""
    key = (sigma, A, k, gamma, Q, steps, T)
    if key not in _policy_cache:
        # Check deadline
        if time.time() > _DEADLINE - 10:
            # Not enough time, return a simple heuristic
            return None
        _policy_cache[key] = _solve_regime(sigma, A, k, gamma, Q, steps, T)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """Inventory-aware quoting using pre-computed HJB policy."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    steps = params['steps']
    T = params['T']
    
    # Determine current step
    dt = T / steps
    n = int(round(t / dt))
    n = min(max(n, 0), steps - 1)
    
    policy = _get_policy(sigma, A, k, gamma, Q, steps, T)
    
    if policy is None:
        # Fallback to heuristic if policy not available
        q = inventory.astype(np.float64)
        base_depth = 1.0 / k
        beta = 0.02
        bid_depth = base_depth + beta * q
        ask_depth = base_depth - beta * q
        min_depth = 0.01
        bid_depth = np.maximum(bid_depth, min_depth)
        ask_depth = np.maximum(ask_depth, min_depth)
        bid_depth[inventory >= Q] = np.inf
        ask_depth[inventory <= -Q] = np.inf
        return bid_depth, ask_depth
    
    # Get depths from policy
    q_idx = inventory + Q  # Convert inventory to index
    bid_depth = policy[n, q_idx, 0]
    ask_depth = policy[n, q_idx, 1]
    
    # Ensure valid depths
    bid_depth = np.where(bid_depth < 0.001, 0.001, bid_depth)
    ask_depth = np.where(ask_depth < 0.001, 0.001, ask_depth)
    
    # At boundaries, no quote on the side that would push inventory further
    bid_depth[inventory >= Q] = np.inf
    ask_depth[inventory <= -Q] = np.inf
    
    return bid_depth, ask_depth
# EVOLVE-BLOCK-END
