"""
Optimal market making via pre-computed DP policy lookup.
The state space is (step, inventory) with 1000 steps and 61 inventory levels (-30..30).
We solve the Bellman equation backwards to find optimal bid/ask depths for each state.
Results are cached per regime so only the first call pays the DP cost.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 325.0  # safety margin before 330s kill


def _compute_optimal_policy(params):
    """
    Compute optimal bid/ask depths for all (step, inventory) states via backward DP.
    
    Returns:
        bid_table: array of shape (steps, 2*Q+1) - optimal bid depth for each state
        ask_table: array of shape (steps, 2*Q+1) - optimal ask depth for each state
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    T = params['T']
    n_steps = params.get('steps', 1000)
    dt = T / n_steps
    
    n_inv = 2 * Q + 1
    inv_offset = Q  # inventory index: -Q -> 0, 0 -> Q, Q -> 2Q
    
    # Depth grid for search (vectorized)
    # Use a fine grid to approximate optimal depths
    max_d = 10.0 / k
    n_depths = 200
    depth_grid = np.linspace(0.0, max_d, n_depths)
    # Also include np.inf as "no quote" option
    all_depths = np.concatenate([depth_grid, [np.inf]])
    n_opts = len(all_depths)
    
    # Precompute fill probabilities for all depths
    # p_fill(d) = 1 - exp(-A * exp(-k*d) * dt)
    # For d = inf, p_fill = 0
    p_fill = np.zeros(n_opts)
    finite_mask = np.isfinite(all_depths)
    if np.any(finite_mask):
        exp_term = A * np.exp(-k * all_depths[finite_mask]) * dt
        p_fill[finite_mask] = 1.0 - np.exp(-exp_term)
    
    # Expected gain from quoting at depth d: d * p_fill(d)
    # For d = inf, gain = 0
    gain = np.zeros(n_opts)
    gain[finite_mask] = all_depths[finite_mask] * p_fill[finite_mask]
    
    # DP tables
    V = np.zeros((n_steps + 1, n_inv))  # value function
    bid_table = np.zeros((n_steps, n_inv))
    ask_table = np.zeros((n_steps, n_inv))
    
    # Terminal condition: V[n_steps, :] = 0 (no terminal charge)
    
    # Backward induction
    for n in range(n_steps - 1, -1, -1):
        # Check time limit
        if time.time() - _START_TIME > TIME_LIMIT - 5.0:
            # Return what we have so far (remaining steps use heuristic)
            # Fill remaining with a simple heuristic
            for nn in range(n - 1, -1, -1):
                for ii in range(n_inv):
                    q_val = ii - inv_offset
                    d0 = 1.0 / k
                    bid_table[nn, ii] = d0
                    ask_table[nn, ii] = d0
            # Fix boundary conditions
            bid_table[:, 2 * Q] = np.inf  # q = Q, no bid
            ask_table[:, 0] = np.inf       # q = -Q, no ask
            return bid_table, ask_table
        
        # Inventory values for this step
        inv_vals = np.arange(n_inv) - inv_offset  # -Q, ..., Q
        
        # Penalty for current inventory: (gamma/2) * sigma^2 * q^2 * dt
        penalty = (gamma / 2.0) * sigma * sigma * inv_vals * inv_vals * dt
        
        # For each inventory state, compute optimal bid and ask
        for ii in range(n_inv):
            q_val = ii - inv_offset
            
            # No bid if q = Q, no ask if q = -Q
            if q_val >= Q:
                bid_table[n, ii] = np.inf
                bid_gain = 0.0
                bid_prob = 0.0
            else:
                # Choose bid depth to maximize: gain_bid + expected future value
                # If bid fills: inventory increases by 1, gain = bid_depth
                # If no bid fill: inventory stays same
                # E[V_next | bid] = p_bid * V[n+1, ii+1] + (1-p_bid) * V[n+1, ii]
                # Total = bid_gain + E[V_next] - penalty
                # We maximize over bid_depth
                
                # Vectorized: for each bid depth option
                if q_val < Q:
                    p_b = p_fill
                    # Future value: p_b * V[n+1, ii+1] + (1-p_b) * V[n+1, ii]
                    future_val_b = p_b * V[n + 1, ii + 1] + (1.0 - p_b) * V[n + 1, ii]
                    total_b = gain + future_val_b
                    best_b_idx = np.argmax(total_b)
                    bid_table[n, ii] = all_depths[best_b_idx]
                    bid_gain = gain[best_b_idx]
                    bid_prob = p_fill[best_b_idx]
                else:
                    bid_table[n, ii] = np.inf
                    bid_gain = 0.0
                    bid_prob = 0.0
            
            if q_val <= -Q:
                ask_table[n, ii] = np.inf
                ask_gain = 0.0
                ask_prob = 0.0
            else:
                # Choose ask depth to maximize: gain_ask + expected future value
                # If ask fills: inventory decreases by 1, gain = ask_depth
                # If no ask fill: inventory stays same
                if q_val > -Q:
                    p_a = p_fill
                    # Future value: p_a * V[n+1, ii-1] + (1-p_a) * V[n+1, ii]
                    future_val_a = p_a * V[n + 1, ii - 1] + (1.0 - p_a) * V[n + 1, ii]
                    total_a = gain + future_val_a
                    best_a_idx = np.argmax(total_a)
                    ask_table[n, ii] = all_depths[best_a_idx]
                    ask_gain = gain[best_a_idx]
                    ask_prob = p_fill[best_a_idx]
                else:
                    ask_table[n, ii] = np.inf
                    ask_gain = 0.0
                    ask_prob = 0.0
            
            # Joint probability: bid and ask fills are independent
            # Outcomes:
            # 1. Both fill: prob = p_b * p_a, inv change = 0, gain = bid_gain + ask_gain
            # 2. Bid only: prob = p_b * (1-p_a), inv change = +1, gain = bid_gain
            # 3. Ask only: prob = (1-p_b) * p_a, inv change = -1, gain = ask_gain
            # 4. Neither: prob = (1-p_b) * (1-p_a), inv change = 0, gain = 0
            
            p_both = bid_prob * ask_prob
            p_bid_only = bid_prob * (1.0 - ask_prob)
            p_ask_only = (1.0 - bid_prob) * ask_prob
            p_neither = (1.0 - bid_prob) * (1.0 - ask_prob)
            
            # Expected future value
            if q_val + 1 <= Q:
                fv_bid = V[n + 1, ii + 1]
            else:
                fv_bid = 0.0  # shouldn't happen due to boundary
            
            if q_val - 1 >= -Q:
                fv_ask = V[n + 1, ii - 1]
            else:
                fv_ask = 0.0  # shouldn't happen due to boundary
            
            fv_none = V[n + 1, ii]
            
            # Both fill: net inventory change = 0
            fv_both = V[n + 1, ii]
            
            exp_future = p_both * fv_both + p_bid_only * fv_bid + p_ask_only * fv_ask + p_neither * fv_none
            
            # Current step reward: earned depths - penalty
            current_reward = bid_gain + ask_gain - penalty[ii]
            
            V[n, ii] = current_reward + exp_future
    
    # Fix boundary conditions (should already be handled, but ensure)
    bid_table[:, 2 * Q] = np.inf  # q = Q
    ask_table[:, 0] = np.inf       # q = -Q
    
    return bid_table, ask_table


def quote(t, inventory, mid, params):
    """
    Return optimal bid/ask depths using pre-computed DP policy.
    """
    T = params["T"]
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    Q = params["Q"]
    n_steps = params.get("steps", 1000)
    dt = T / n_steps
    
    # Create cache key
    cache_key = (sigma, A, k, gamma, Q, T, n_steps)
    
    if cache_key not in _policy_cache:
        bid_table, ask_table = _compute_optimal_policy(params)
        _policy_cache[cache_key] = (bid_table, ask_table)
    else:
        bid_table, ask_table = _policy_cache[cache_key]
    
    # Determine current step
    n = int(round(t / dt))
    n = np.clip(n, 0, n_steps - 1)
    
    # Map inventory to index: -Q -> 0, 0 -> Q, Q -> 2Q
    inv_idx = inventory + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Look up depths
    bid_depths = bid_table[n, inv_idx].astype(np.float64)
    ask_depths = ask_table[n, inv_idx].astype(np.float64)
    
    # Ensure valid values
    min_valid = -3.0 / k
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths >= min_valid), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths >= min_valid), ask_depths, np.inf)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
