import numpy as np

_dp_cache = {}

def _solve_dp(sigma, A, k, gamma, T, steps, Q):
    key = (sigma, A, k, gamma)
    if key in _dp_cache:
        return _dp_cache[key]
        
    # Grid parameters
    # Optimal depth is typically O(1/k). 
    # We use a grid from 0 to 4/k to be safe.
    max_d = 4.0 / k
    G = 200
    ds = np.linspace(0, max_d, G)
    
    V_next = np.zeros(2*Q + 1)
    opt_bid = np.zeros((steps, 2*Q + 1))
    opt_ask = np.zeros((steps, 2*Q + 1))
    
    dt = T / steps
    q_vals = np.arange(-Q, Q+1)
    M = len(q_vals)
    
    # Pre-compute V_pad indices
    idx = q_vals + Q # 0..2Q
    
    for n in range(steps - 1, -1, -1):
        # q shape (M, 1, 1)
        q = q_vals.reshape(-1, 1, 1)
        
        # Grids
        db_grid = ds.reshape(1, -1, 1)
        da_grid = ds.reshape(1, 1, -1)
        
        # Probabilities
        p_b_raw = 1.0 - np.exp(-A * np.exp(-k * db_grid) * dt)
        p_a_raw = 1.0 - np.exp(-A * np.exp(-k * da_grid) * dt)
        
        # Boundary masks
        # q == Q -> no bid
        # q == -Q -> no ask
        mask_b = (q_vals != Q).reshape(-1, 1, 1).astype(float)
        mask_a = (q_vals != -Q).reshape(-1, 1, 1).astype(float)
        
        p_b = p_b_raw * mask_b
        p_a = p_a_raw * mask_a
        
        # Reward
        cost = (gamma / 2.0) * (sigma**2) * (q**2) * dt
        reward_depths = p_b * db_grid + p_a * da_grid
        
        # Expected Value
        V_pad = np.zeros(2*Q + 3)
        V_pad[1:2*Q+2] = V_next
        
        V_q = V_pad[idx + 1]
        V_q_plus = V_pad[idx + 2]
        V_q_minus = V_pad[idx]
        
        V_q_b = V_q.reshape(-1, 1, 1)
        V_q_plus_b = V_q_plus.reshape(-1, 1, 1)
        V_q_minus_b = V_q_minus.reshape(-1, 1, 1)
        
        p_up = p_b * (1.0 - p_a)
        p_down = (1.0 - p_b) * p_a
        p_same = (1.0 - p_b) * (1.0 - p_a) + p_b * p_a
        
        EV = p_up * V_q_plus_b + p_down * V_q_minus_b + p_same * V_q_b
        
        Objective = reward_depths - cost + EV
        
        flat_obj = Objective.reshape(M, -1)
        max_indices = np.argmax(flat_obj, axis=1)
        
        db_idx = max_indices // G
        da_idx = max_indices % G
        
        opt_bid[n] = ds[db_idx]
        opt_ask[n] = ds[da_idx]
        
        V_next = flat_obj[np.arange(M), max_indices]
        
    _dp_cache[key] = (opt_bid, opt_ask)
    return _dp_cache[key]

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    opt_bid, opt_ask = _solve_dp(sigma, A, k, gamma, T, steps, Q)
    
    dt = T / steps
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    inv_indices = inventory + Q
    
    bid_depths = opt_bid[n, inv_indices]
    ask_depths = opt_ask[n, inv_indices]
    
    return bid_depths, ask_depths
