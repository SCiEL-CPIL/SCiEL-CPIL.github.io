"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed policies per regime
_policy_cache = {}


def _compute_optimal_depths(params):
    """
    Compute optimal bid/ask depths via vectorized dynamic programming.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1)

    # Value function
    V_next = np.zeros(n_states)

    # Store optimal actions
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))

    # Precompute cost
    cost = 0.5 * gamma * sigma**2 * (q_values**2) * dt

    # Precompute depth grid for optimization
    # We'll use a fine grid to find optimal depths
    d_grid = np.linspace(0, 20.0, 2000)
    d_grid = np.maximum(d_grid, 1e-10)

    # Backward induction
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_states)
        
        # For each state, compute optimal bid and ask depths
        # Bid: for states 0 to n_states-2
        # Ask: for states 1 to n_states-1
        
        # Compute delta values
        # For bid: delta_b[i] = V_next[i+1] - V_next[i] for i in 0..n_states-2
        delta_b = np.zeros(n_states - 1)
        delta_b[:] = V_next[1:] - V_next[:-1]
        
        # For ask: delta_a[i] = V_next[i-1] - V_next[i] for i in 1..n_states-1
        delta_a = np.zeros(n_states - 1)
        delta_a[:] = V_next[:-1] - V_next[1:]
        
        # For each possible delta, find optimal depth
        # h(d) = p(d) * (d + delta)
        # where p(d) = 1 - exp(-A * exp(-k*d) * dt)
        
        # Vectorize: for each delta, compute h(d) over d_grid
        # delta_b has shape (n_states-1,), d_grid has shape (2000,)
        # We want h[b, d] = p(d_grid[d]) * (d_grid[d] + delta_b[b])
        
        # Compute p for all d in d_grid
        lam_grid = A * np.exp(-k * d_grid) * dt
        p_grid = 1.0 - np.exp(-lam_grid)
        
        # For bid depths
        # h_b[b, d] = p_grid[d] * (d_grid[d] + delta_b[b])
        # This is a (n_states-1, 2000) array
        h_b = p_grid[None, :] * (d_grid[None, :] + delta_b[:, None])
        # Find argmax over d for each b
        b_idx = np.argmax(h_b, axis=1)
        opt_bid[n, :-1] = d_grid[b_idx]
        
        # For ask depths
        h_a = p_grid[None, :] * (d_grid[None, :] + delta_a[:, None])
        a_idx = np.argmax(h_a, axis=1)
        opt_ask[n, 1:] = d_grid[a_idx]
        
        # Boundary: no bid at q=Q, no ask at q=-Q
        opt_bid[n, -1] = np.inf
        opt_ask[n, 0] = np.inf
        
        # Compute fill probabilities
        p_b = np.zeros(n_states)
        p_a = np.zeros(n_states)
        
        # For states 0 to n_states-2
        lam_b = A * np.exp(-k * opt_bid[n, :-1]) * dt
        p_b[:-1] = 1.0 - np.exp(-lam_b)
        
        # For states 1 to n_states-1
        lam_a = A * np.exp(-k * opt_ask[n, 1:]) * dt
        p_a[1:] = 1.0 - np.exp(-lam_a)
        
        # Compute expected value
        # V_curr[i] = p_b[i]*b_opt[i] + p_a[i]*a_opt[i] 
        #            + p_b[i]*(1-p_a[i])*V_next[i+1] 
        #            + (1-p_b[i])*p_a[i]*V_next[i-1] 
        #            + (1-p_b[i]-p_a[i]+p_b[i]*p_a[i])*V_next[i] 
        #            - cost[i]
        
        # Handle boundaries
        V_next_shift_up = np.zeros(n_states)
        V_next_shift_up[:-1] = V_next[1:]
        # V_next_shift_up[-1] = 0 (no bid at Q)
        
        V_next_shift_down = np.zeros(n_states)
        V_next_shift_down[1:] = V_next[:-1]
        # V_next_shift_down[0] = 0 (no ask at -Q)
        
        # Earnings
        earn_b = p_b * opt_bid[n]
        earn_a = p_a * opt_ask[n]
        # Replace inf with 0
        earn_b[~np.isfinite(earn_b)] = 0.0
        earn_a[~np.isfinite(earn_a)] = 0.0
        
        # Probabilities
        prob_b_only = p_b * (1.0 - p_a)
        prob_a_only = (1.0 - p_b) * p_a
        prob_none = 1.0 - p_b - p_a + p_b * p_a
        
        v_next_expected = prob_b_only * V_next_shift_up + prob_a_only * V_next_shift_down + prob_none * V_next
        
        V_curr = earn_b + earn_a + v_next_expected - cost
        
        V_next = V_curr

    return opt_bid, opt_ask


def _get_policy(params):
    """Get cached policy or compute it."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _policy_cache:
        _policy_cache[key] = _compute_optimal_depths(params)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Quote depths based on precomputed optimal policy.
    """
    # Deadline guard
    start_time = time.time()
    
    steps = params['steps']
    Q = params['Q']
    
    # Get or compute policy
    try:
        opt_bid, opt_ask = _get_policy(params)
    except Exception:
        # Fallback to seed behavior
        depth = np.full(len(inventory), 1.0 / params['k'])
        return depth, depth.copy()
    
    # Determine current step index
    dt = params['T'] / steps
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    # Map inventory to index
    idx = inventory + Q
    idx = np.clip(idx, 0, 2*Q)
    
    # Get depths
    bid_depths = opt_bid[n, idx]
    ask_depths = opt_ask[n, idx]
    
    # Ensure valid outputs
    # If depth is inf, it means no quote.
    # If depth is 0, it means quote at mid.
    # Clip to valid range: >= 0 and finite or inf
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < 0), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < 0), np.inf, ask_depths)
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
