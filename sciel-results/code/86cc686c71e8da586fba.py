import numpy as np

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    n_steps = steps
    q_min = -Q
    q_max = Q
    n_q = q_max - q_min + 1
    
    V = np.zeros((n_steps + 1, n_q))
    opt_bid = np.zeros((n_steps, n_q))
    opt_ask = np.zeros((n_steps, n_q))
    
    max_depth = 3.0 / k
    n_depth = 30
    depth_grid = np.linspace(0.005, max_depth, n_depth)
    
    p_fill = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    q_vals = np.arange(q_min, q_max + 1)
    hold_costs = 0.5 * gamma * sigma**2 * dt * (q_vals**2)
    
    for n in range(n_steps - 1, -1, -1):
        V_next = V[n + 1]
        
        for q_idx in range(n_q):
            q = q_min + q_idx
            
            if q < q_max:
                pb = p_fill
            else:
                pb = np.zeros(n_depth)
            
            if q > q_min:
                pa = p_fill
            else:
                pa = np.zeros(n_depth)
            
            pb_grid, pa_grid = np.meshgrid(pb, pa, indexing='ij')
            bid_grid, ask_grid = np.meshgrid(depth_grid, depth_grid, indexing='ij')
            
            imm = pb_grid * bid_grid + pa_grid * ask_grid - hold_costs[q_idx]
            
            q_bid = min(q + 1, q_max)
            q_ask = max(q - 1, q_min)
            q_bid_idx = q_bid - q_min
            q_ask_idx = q_ask - q_min
            
            p_both = pb_grid * pa_grid
            p_bid_only = pb_grid * (1.0 - pa_grid)
            p_ask_only = (1.0 - pb_grid) * pa_grid
            p_none = (1.0 - pb_grid) * (1.0 - pa_grid)
            
            future = (p_both * V_next[q_bid_idx] +
                     p_bid_only * V_next[q_bid_idx] +
                     p_ask_only * V_next[q_ask_idx] +
                     p_none * V_next[q_idx])
            
            total = imm + future
            idx = np.argmax(total)
            i, j = np.unravel_index(idx, total.shape)
            
            V[n, q_idx] = total[i, j]
            opt_bid[n, q_idx] = depth_grid[i]
            opt_ask[n, q_idx] = depth_grid[j]
    
    n = int(round(t / dt))
    n = max(0, min(n, n_steps - 1))
    
    bid_depths = np.empty(len(inventory))
    ask_depths = np.empty(len(inventory))
    
    for i in range(len(inventory)):
        q = int(inventory[i])
        q_idx = q - q_min
        if q_idx < 0 or q_idx >= n_q:
            bid_depths[i] = np.inf
            ask_depths[i] = np.inf
        else:
            bid_depths[i] = opt_bid[n, q_idx]
            ask_depths[i] = opt_ask[n, q_idx]
    
    return bid_depths, ask_depths
