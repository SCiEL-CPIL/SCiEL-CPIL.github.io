"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import OrderedDict

# Cache for precomputed optimal policies
_policy_cache = OrderedDict()


def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute the optimal bid/ask depth for every (step, inventory) pair
    via backward dynamic programming.
    
    Returns:
        bid_depths: array of shape (steps, 2*Q+1)
        ask_depths: array of shape (steps, 2*Q+1)
    """
    dt = T / steps
    n_inventory = 2 * Q + 1
    
    # V_next[q_idx] where q_idx = q + Q
    V_next = np.zeros(n_inventory)
    
    bid_depths = np.zeros((steps, n_inventory))
    ask_depths = np.zeros((steps, n_inventory))
    
    # Grid of depth candidates
    # For small depths, use finer grid
    # exp(-k*d) for d in [0, 5/k]
    d_max = 5.0 / k
    n_d = 80
    d_grid = np.linspace(0.0, d_max, n_d)
    
    for n in range(steps - 1, -1, -1):
        # V_curr[q_idx] = max over bid_depth, ask_depth of expected value
        # = bid_depth * p_bid + ask_depth * p_ask - (gamma/2)*sigma^2*q^2*dt + V_next[q_next]
        # where q_next = q + 1 (bid hit) or q - 1 (ask hit)
        
        V_curr = np.zeros(n_inventory)
        
        for q_idx in range(n_inventory):
            q = q_idx - Q
            
            # Holding cost
            holding_cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            best_val = -np.inf
            best_bid = d_max
            best_ask = d_max
            
            # Determine valid next inventory states
            # Bid hit: q -> q+1 (invalid if q == Q)
            # Ask hit: q -> q-1 (invalid if q == -Q)
            
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            if can_bid:
                V_next_bid = V_next[q_idx + 1]
            else:
                V_next_bid = 0.0  # no transition, stays at boundary (but no fill)
            
            if can_ask:
                V_next_ask = V_next[q_idx - 1]
            else:
                V_next_ask = 0.0
            
            # Evaluate all combinations
            # p_bid(d) = 1 - exp(-A * exp(-k*d) * dt)
            # p_ask(d) = 1 - exp(-A * exp(-k*d) * dt)
            
            for i in range(n_d):
                d_bid = d_grid[i]
                p_bid = 1.0 - np.exp(-A * np.exp(-k * d_bid) * dt)
                
                if not can_bid:
                    p_bid = 0.0
                
                for j in range(n_d):
                    d_ask = d_grid[j]
                    p_ask = 1.0 - np.exp(-A * np.exp(-k * d_ask) * dt)
                    
                    if not can_ask:
                        p_ask = 0.0
                    
                    # Expected value
                    # If bid hit: +d_bid, q_next = q+1
                    # If ask hit: +d_ask, q_next = q-1
                    # Both can happen? The model says "independently" but a fill moves inventory one unit.
                    # "A fill earns its depth and moves inventory one unit"
                    # "bid hit with probability ... ask likewise, independently"
                    # This implies both could hit. If both hit, what happens to inventory?
                    # Typically in market making, if both hit, inventory doesn't change (net 0) or we handle it as separate events.
                    # "moves inventory one unit" - likely net change.
                    # Let's assume:
                    # - Bid only: q+1, earn d_bid
                    # - Ask only: q-1, earn d_ask
                    # - Both: q unchanged (net 0), earn d_bid + d_ask? Or just one?
                    # Usually, if both cross, you buy one and sell one. Inventory stays same. Profit is sum of depths.
                    # - Neither: q unchanged, earn 0
                    
                    # Probabilities:
                    p_bid_only = p_bid * (1.0 - p_ask)
                    p_ask_only = (1.0 - p_bid) * p_ask
                    p_both = p_bid * p_ask
                    p_none = (1.0 - p_bid) * (1.0 - p_ask)
                    
                    val = 0.0
                    if can_bid and can_ask:
                        # Both can change inventory
                        # Bid hit: q+1
                        # Ask hit: q-1
                        # Both: q unchanged (net 0)
                        # None: q unchanged
                        
                        val = (p_bid_only * (d_bid + V_next_bid) +
                               p_ask_only * (d_ask + V_next_ask) +
                               p_both * (d_bid + d_ask + V_next[q_idx]) +
                               p_none * V_next[q_idx])
                    elif can_bid:
                        # Only bid can hit
                        val = p_bid * (d_bid + V_next_bid) + (1.0 - p_bid) * V_next[q_idx]
                    elif can_ask:
                        # Only ask can hit
                        val = p_ask * (d_ask + V_next_ask) + (1.0 - p_ask) * V_next[q_idx]
                    else:
                        val = V_next[q_idx]
                    
                    val -= holding_cost
                    
                    if val > best_val:
                        best_val = val
                        best_bid = d_bid
                        best_ask = d_ask
            
            V_curr[q_idx] = best_val
            bid_depths[n, q_idx] = best_bid
            ask_depths[n, q_idx] = best_ask
        
        V_next = V_curr
    
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """
    Quote both sides using precomputed optimal policy.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Create cache key
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _policy_cache:
        bid_depths, ask_depths = _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
        _policy_cache[key] = (bid_depths, ask_depths)
        
        # Limit cache size
        if len(_policy_cache) > 10:
            _policy_cache.popitem(last=False)
    else:
        bid_depths, ask_depths = _policy_cache[key]
    
    # Determine current step
    dt = T / steps
    n = int(round(t / dt))
    n = np.clip(n, 0, steps - 1)
    
    # Map inventory to index
    inv_idx = inventory + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Get depths for this step and inventory
    bid_depths_step = bid_depths[n]
    ask_depths_step = ask_depths[n]
    
    result_bid = bid_depths_step[inv_idx].astype(np.float64)
    result_ask = ask_depths_step[inv_idx].astype(np.float64)
    
    return result_bid, result_ask


# EVOLVE-BLOCK-END
