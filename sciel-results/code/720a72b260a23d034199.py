"""
Strategy: Pre-compute the stationary optimal policy using Value Iteration.
This avoids the 1000-step forward/backward DP, allowing for a much finer 
depth grid resolution (N_grid > 1000) within the time limit, which reduces
discretization error in the optimal depth selection.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for pre-computed optimal policies
_policy_cache = {}
_start_time = time.time()
TIME_LIMIT = 320.0  # Safety margin

def _compute_stationary_policy(params):
    """
    Compute the optimal bid and ask depths for all inventory states q using 
    stationary Value Iteration.
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
    q_values = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # We store the best depth for each state q
    # Since the policy is stationary, we don't need time dimension
    # But we need to return arrays of shape (4000,) in quote(), 
    # which are indexed by time step n. 
    # If the policy is truly stationary, bid_depths[n, q] = pi_bid(q) for all n.
    
    # Grid resolution: Use a much finer grid since we only run ~100 iterations
    # instead of 1000.
    max_depth = 3.0
    # Adaptive grid size based on time budget? 
    # Let's aim for high resolution. 
    # 100 iterations * 61 states * (N_grid^2) operations.
    # If N_grid = 500, 500^2 = 250k. 100 * 61 * 250k = 1.5B ops. Too slow in Python/NumPy overhead?
    # NumPy vectorizes the N_grid^2 part. 
    # The bottleneck is the (n_states, n_grid, n_grid) array construction.
    # 61 * 500 * 500 * 8 bytes = 122 MB per step. 100 steps = 12 GB traffic. Might be slow.
    # Let's balance. N_grid = 200 is safe. N_grid = 300 is current. 
    # Let's try N_grid = 400.
    
    n_grid = 400
    grid = np.linspace(0, max_depth, n_grid)
    
    # Precompute probabilities
    exp_term = np.exp(-k * grid)
    p_grid = 1.0 - np.exp(-A * exp_term * dt)
    one_minus_p = 1.0 - p_grid
    
    # Precompute cost for each state
    cost_per_state = 0.5 * gamma * sigma**2 * q_values**2 * dt  # (n_states,)
    
    # Precompute broadcast arrays for vectorization
    grid_b = grid[None, :, None]   # (1, n_grid, 1)
    grid_a = grid[None, None, :]   # (1, 1, n_grid)
    pb_col = p_grid[None, :, None]         # (1, n_grid, 1)
    pa_row = p_grid[None, None, :]         # (1, 1, n_grid)
    ompb_col = one_minus_p[None, :, None]  # (1, n_grid, 1)
    ompa_row = one_minus_p[None, None, :]  # (1, 1, n_grid)
    
    V = np.zeros(n_states)
    
    opt_bid_q = np.zeros(n_states)
    opt_ask_q = np.zeros(n_states)
    
    max_iter = 200
    tol = 1e-6
    
    for it in range(max_iter):
        if time.time() - _start_time > TIME_LIMIT:
            break
            
        v_q = V  # (n_states,)
        
        # v_qp: V[i+1] for i < n_states-1, else 0 (no bid at q=Q)
        v_qp = np.zeros(n_states)
        v_qp[:-1] = V[1:]
        
        # v_qm: V[i-1] for i > 0, else 0 (no ask at q=-Q)
        v_qm = np.zeros(n_states)
        v_qm[1:] = V[:-1]
        
        # Build 3D total: (n_states, n_grid, n_grid)
        # t1: no fills
        t1 = v_q[:, None, None] * ompb_col * ompa_row
        
        # t2: bid fill only
        t2 = v_qp[:, None, None] * pb_col * ompa_row + grid_b * pb_col * ompa_row
        
        # t3: ask fill only
        t3 = v_qm[:, None, None] * ompb_col * pa_row + grid_a * ompb_col * pa_row
        
        # t4: both fills
        t4 = v_q[:, None, None] * pb_col * pa_row + (grid_b + grid_a) * pb_col * pa_row
        
        total = t1 + t2 + t3 + t4 - cost_per_state[:, None, None]
        
        # Find argmax over last two axes for each state
        total_flat = total.reshape(n_states, -1)
        max_idx = np.argmax(total_flat, axis=1)
        i = max_idx // n_grid
        j = max_idx % n_grid
        
        new_V = total_flat[np.arange(n_states), max_idx]
        
        # Update policy
        opt_bid_q = grid[i]
        opt_ask_q = grid[j]
        
        # Check convergence
        diff = np.max(np.abs(new_V - V))
        V = new_V
        
        if diff < tol:
            break
            
    # Construct the full policy arrays of shape (steps, n_states)
    # Since it's stationary, we just tile the q-based policy
    opt_bid = np.tile(opt_bid_q[None, :], (steps, 1))
    opt_ask = np.tile(opt_ask_q[None, :], (steps, 1))
    
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    """
    Look up optimal depths from the pre-computed stationary policy.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        bid_depths, ask_depths = _compute_stationary_policy(params)
        _policy_cache[key] = (bid_depths, ask_depths)
    else:
        bid_depths, ask_depths = _policy_cache[key]
    
    n = int(np.round(t * params['steps']))
    if n >= params['steps']:
        n = params['steps'] - 1
        
    indices = inventory + params['Q']
    
    bids = bid_depths[n, indices]
    asks = ask_depths[n, indices]
    
    return bids, asks

# EVOLVE-BLOCK-END
