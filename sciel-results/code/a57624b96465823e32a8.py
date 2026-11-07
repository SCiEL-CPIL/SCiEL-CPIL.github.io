import numpy as np

def _precompute_tables(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    N = steps
    num_states = 2 * Q + 1
    
    V_next = np.zeros(num_states)
    opt_bid = np.zeros((N, num_states))
    opt_ask = np.zeros((N, num_states))
    
    # Use a finer grid for better accuracy
    num_grid = 200
    
    # Determine max depth
    d_max = np.log(max(A * dt / 1e-4, 1.0)) / k
    d_max = np.clip(d_max, 1.0, 20.0)
    
    d_grid = np.linspace(0, d_max, num_grid)
    
    # Precompute 1D probabilities
    p_grid_1d = 1 - np.exp(-A * np.exp(-k * d_grid) * dt)
    
    # Create 2D grid
    d_b_grid, d_a_grid = np.meshgrid(d_grid, d_grid, indexing='ij')
    d_b_flat = d_b_grid.flatten()
    d_a_flat = d_a_grid.flatten()
    
    p_b_flat = 1 - np.exp(-A * np.exp(-k * d_b_flat) * dt)
    p_a_flat = 1 - np.exp(-A * np.exp(-k * d_a_flat) * dt)
    
    C_grid = p_b_flat * (1 - p_a_flat)
    D_grid = (1 - p_b_flat) * p_a_flat
    E_grid = 1 - p_b_flat - p_a_flat + 2 * p_b_flat * p_a_flat
    R_grid = p_b_flat * d_b_flat + p_a_flat * d_a_flat
    
    q_vals = np.arange(-Q, Q + 1)
    costs = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    for n in range(N - 1, -1, -1):
        Delta_plus = np.zeros(num_states)
        Delta_minus = np.zeros(num_states)
        
        Delta_plus[:-1] = V_next[1:] - V_next[:-1]
        Delta_minus[1:] = V_next[:-1] - V_next[1:]
        
        Val_matrix = R_grid[:, None] + C_grid[:, None] * Delta_plus[None, :] + D_grid[:, None] * Delta_minus[None, :] + E_grid[:, None] * V_next[None, :]
        Val_matrix -= costs[None, :]
        
        best_idx = np.argmax(Val_matrix, axis=0)
        V_curr = Val_matrix[best_idx, np.arange(num_states)]
        
        opt_d_b = d_b_flat[best_idx]
        opt_d_a = d_a_flat[best_idx]
        
        if num_states > 1:
            # q = Q (index -1)
            q_idx = num_states - 1
            vals_qQ = p_grid_1d * (d_grid + Delta_minus[q_idx]) + V_next[q_idx] - costs[q_idx]
            best_a_idx = np.argmax(vals_qQ)
            opt_d_a[q_idx] = d_grid[best_a_idx]
            opt_d_b[q_idx] = np.inf
            V_curr[q_idx] = vals_qQ[best_a_idx]
            
            # q = -Q (index 0)
            q_idx = 0
            vals_qmQ = p_grid_1d * (d_grid + Delta_plus[q_idx]) + V_next[q_idx] - costs[q_idx]
            best_b_idx = np.argmax(vals_qmQ)
            opt_d_b[q_idx] = d_grid[best_b_idx]
            opt_d_a[q_idx] = np.inf
            V_curr[q_idx] = vals_qmQ[best_b_idx]
            
        opt_bid[n] = opt_d_b
        opt_ask[n] = opt_d_a
        V_next = V_curr
        
    return opt_bid, opt_ask

_cache = {}

def quote(t, inventory, mid, params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _cache:
        _cache[key] = _precompute_tables(params)
    
    opt_bid, opt_ask = _cache[key]
    
    steps = params['steps']
    T = params['T']
    dt = T / steps
    Q = params['Q']
    
    n = int(t / dt)
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    idx = inventory + Q
    idx = np.clip(idx, 0, 2 * Q)
    
    bid_depths = opt_bid[n, idx]
    ask_depths = opt_ask[n, idx]
    
    return bid_depths, ask_depths
