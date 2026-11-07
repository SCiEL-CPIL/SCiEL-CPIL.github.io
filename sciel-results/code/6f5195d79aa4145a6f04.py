"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0

def _get_dp_tables(params):
    """Compute optimal bid/ask depth tables via dynamic programming with vectorized grid search."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Value table: V[n, q] for n=0..steps, q=-Q..Q
    # V[steps][q] = 0 (no terminal charge)
    V = np.zeros((steps + 1, 2 * Q + 1))
    
    # Optimal depth tables
    opt_bid = np.full((steps, 2 * Q + 1), np.inf)
    opt_ask = np.full((steps, 2 * Q + 1), np.inf)
    
    # Precompute risk penalty
    q_indices = np.arange(-Q, Q + 1)
    risk = (gamma / 2.0) * sigma**2 * q_indices**2 * dt
    
    # Grid for depth search
    # We need to cover the range where p(d) is significant and d+Delta changes sign
    # p(d) decays as exp(-k d). Significant for d < 5/k.
    # Let's use a grid from 0 to 10/k to be safe.
    d_max = 10.0 / k
    num_points = 500
    D_grid = np.linspace(0.0, d_max, num_points)
    
    # Precompute fill probabilities for the grid
    # p(d) = 1 - exp(-A * exp(-k * d) * dt)
    exp_term = np.exp(-k * D_grid)
    p_grid = 1.0 - np.exp(-A * exp_term * dt)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check time budget
        if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            # If we run out of time, we can't compute more. 
            # But we need to return something. 
            # The current V is the best we have so far (which is 0 for future steps if we abort early? 
            # No, we are filling backwards. If we abort, the remaining steps will have default inf depths and V=0?
            # Actually, if we abort, the cache will be incomplete. 
            # Better to just let it run if it's fast. 
            # With vectorization, it should be very fast.
            pass
        
        V_next = V[n + 1]
        
        # Calculate Delta V for bid and ask
        # Bid: q -> q+1. Delta = V[n+1, q+1] - V[n+1, q]
        # q goes from -Q to Q-1. For q=Q, no bid.
        delta_b = np.full(2 * Q + 1, 0.0)
        if Q > 0:
            delta_b[:-1] = V_next[1:] - V_next[:-1]
            delta_b[-1] = 0.0 # Placeholder, will be ignored or handled by mask
        
        # Ask: q -> q-1. Delta = V[n+1, q-1] - V[n+1, q]
        # q goes from -Q+1 to Q. For q=-Q, no ask.
        delta_a = np.full(2 * Q + 1, 0.0)
        if Q > 0:
            delta_a[1:] = V_next[:-1] - V_next[1:]
            delta_a[0] = 0.0 # Placeholder
        
        # Vectorized optimization for Bid
        # Shape: (2Q+1, num_points)
        # Term: p_grid * (D_grid + delta_b[:, None])
        vals_b = p_grid[None, :] * (D_grid[None, :] + delta_b[:, None])
        max_vals_b = np.max(vals_b, axis=1)
        argmax_idx_b = np.argmax(vals_b, axis=1)
        
        # If max value is <= 0, we prefer no quote (inf) because quoting gives negative or zero expected gain
        # Note: If max is 0, quoting is neutral. Inf is also neutral. We choose Inf for safety/simplicity?
        # Actually, if max is 0, it doesn't matter. 
        # But if max < 0, we MUST choose Inf (or d that minimizes loss, but Inf gives 0 loss).
        # So condition: if max_vals_b > 0, use grid depth, else inf.
        
        bid_depths = np.full(2 * Q + 1, np.inf)
        valid_b = max_vals_b > 0
        bid_depths[valid_b] = D_grid[argmax_idx_b[valid_b]]
        
        # Handle boundary conditions: no bid at q=Q
        if Q > 0:
            bid_depths[-1] = np.inf
            delta_b[-1] = 0.0 # Ensure V update is correct (reward 0)
        
        # Vectorized optimization for Ask
        vals_a = p_grid[None, :] * (D_grid[None, :] + delta_a[:, None])
        max_vals_a = np.max(vals_a, axis=1)
        argmax_idx_a = np.argmax(vals_a, axis=1)
        
        ask_depths = np.full(2 * Q + 1, np.inf)
        valid_a = max_vals_a > 0
        ask_depths[valid_a] = D_grid[argmax_idx_a[valid_a]]
        
        # Handle boundary conditions: no ask at q=-Q
        if Q > 0:
            ask_depths[0] = np.inf
            delta_a[0] = 0.0
        
        # Update Value
        # V[n, q] = V[n+1, q] - risk[q] + bid_reward + ask_reward
        # bid_reward is max_vals_b, but if we chose inf, reward is 0.
        # Our max_vals_b is the max over the grid. If the max over grid is < 0, we chose inf, so reward is 0.
        # So effective reward is np.maximum(max_vals_b, 0.0)
        
        bid_rewards = np.maximum(max_vals_b, 0.0)
        ask_rewards = np.maximum(max_vals_a, 0.0)
        
        # Apply boundary masks to rewards (should be 0 anyway if delta was 0 and p>0? 
        # If delta=0, f(d) = p(d)d. Max is > 0. 
        # But we forced depth to inf at boundaries. So reward should be 0.
        if Q > 0:
            bid_rewards[-1] = 0.0
            ask_rewards[0] = 0.0
        
        V[n] = V_next - risk + bid_rewards + ask_rewards
        opt_bid[n] = bid_depths
        opt_ask[n] = ask_depths
    
    _cache[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """Quote optimal depths based on dynamic programming solution."""
    # Deadline guard
    if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Return a safe fallback: no quotes
        n_paths = len(inventory)
        return np.full(n_paths, np.inf), np.full(n_paths, np.inf)
        
    opt_bid, opt_ask = _get_dp_tables(params)
    
    steps = params['steps']
    T = params['T']
    dt = T / steps
    Q = params['Q']
    
    # Current step index
    n = int(t / dt)
    n = min(n, steps - 1)  # clamp to valid range
    
    # Map inventory to indices
    # inventory is int array in [-Q, Q]
    inv_idx = inventory + Q
    
    # Ensure indices are within bounds (safety)
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    bid_depths = opt_bid[n, inv_idx]
    ask_depths = opt_ask[n, inv_idx]
    
    # Safety: ensure valid values
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > 0), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > 0), ask_depths, np.inf)
    
    return bid_depths.astype(np.float64), ask_depths.astype(np.float64)

# EVOLVE-BLOCK-END
