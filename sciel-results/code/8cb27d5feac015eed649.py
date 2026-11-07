import numpy as np
from scipy.optimize import minimize_scalar
import time

# Global cache for precomputed solutions
_CACHE = {}

def _compute_optimal_depths(params):
    """
    Precomputes optimal bid and ask depths for all steps and inventory levels.
    Returns a tuple (opt_bid, opt_ask) where each is an array of shape (steps, 2*Q+1).
    opt_bid[n, q_idx] is the optimal bid depth for step n and inventory q_idx.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Inventory indices: -Q to Q
    num_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    
    # Precompute optimal depths for a range of W values (W = V(q) - V(q+1) or V(q) - V(q-1))
    # W can be positive or negative.
    # Let's estimate bounds.
    # Max spread ~ 1/k. Max time ~ 1000.
    # Let's use a wide range to be safe.
    W_min = -50.0
    W_max = 50.0
    W_grid = np.linspace(W_min, W_max, 2000)
    
    # Function to find optimal depth for a given W
    def get_opt_depth(W, side):
        # side: 'bid' or 'ask' (symmetric in this formulation for depth optimization)
        # Maximize P(d) * (d + W)
        # P(d) = 1 - exp(-A exp(-k d) dt)
        

        # Define objective to minimize (negative of profit)
        def obj(d):
            if d < 0:
                return 1e9
            mu = A * np.exp(-k * d) * dt
            P = 1 - np.exp(-mu)
            return -P * (d + W)
        
        # Bounds for d
        # If W is very negative, optimal d is small.
        # If W is very positive, optimal d is larger.
        # Max useful d: if d > 10/k, P ~ 1. obj ~ -(d+W). Minimized at smallest d.
        # So we search in [0, 10/k]
        d_max = 10.0 / k
        
        res = minimize_scalar(obj, bounds=(0, d_max), method='bounded', options={'xatol': 1e-6})
        return res.x

    # Precompute depth lookup tables
    # We will map W to index in W_grid
    opt_depths_grid = np.zeros_like(W_grid)
    for i, W in enumerate(W_grid):
        opt_depths_grid[i] = get_opt_depth(W, 'bid')
        
    def lookup_depth(W):
        # Clip W to grid range
        idx = np.searchsorted(W_grid, W)
        if idx == 0:
            return opt_depths_grid[0]
        elif idx >= len(W_grid):
            return opt_depths_grid[-1]
        # Interpolate
        w0 = W_grid[idx-1]
        w1 = W_grid[idx]
        d0 = opt_depths_grid[idx-1]
        d1 = opt_depths_grid[idx]
        if w1 == w0:
            return d0
        frac = (W - w0) / (w1 - w0)
        return d0 + frac * (d1 - d0)

    # DP Backward Pass
    V_next = np.zeros(num_q)
    opt_bid = np.full((steps, num_q), np.inf)
    opt_ask = np.full((steps, num_q), np.inf)
    
    # Precompute inventory cost term
    inv_cost = (gamma / 2.0) * sigma**2 * (q_vals**2) * dt
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(num_q)
        
        for i_q, q in enumerate(q_vals):
            # Determine valid sides
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            if can_bid:
                # W_b = V_next(q) - V_next(q+1)
                # q_idx = i_q, q+1_idx = i_q + 1
                W_b = V_next[i_q] - V_next[i_q + 1]
                b_opt = lookup_depth(W_b)
                opt_bid[n, i_q] = b_opt
                
                mu_b = A * np.exp(-k * b_opt) * dt
                P_b = 1 - np.exp(-mu_b)
            else:
                b_opt = np.inf
                P_b = 0.0
                W_b = 0.0
                
            if can_ask:
                # W_a = V_next(q) - V_next(q-1)
                # q_idx = i_q, q-1_idx = i_q - 1
                W_a = V_next[i_q] - V_next[i_q - 1]
                a_opt = lookup_depth(W_a)
                opt_ask[n, i_q] = a_opt
                
                mu_a = A * np.exp(-k * a_opt) * dt
                P_a = 1 - np.exp(-mu_a)
            else:
                a_opt = np.inf
                P_a = 0.0
                W_a = 0.0
            
            # Calculate V_curr(q)
            # V_curr = V_next(q) - Cost + P_b(b + V_next(q) - V_next(q+1)) + P_a(a + V_next(q) - V_next(q-1))
            # Wait, the term in max was P_b(b - (V_next(q+1) - V_next(q))) ...
            # Let's re-verify the recursion:
            # V_n(q) = V_{n+1}(q) - C + max [ P_b(b + W_b) + P_a(a + W_a) ]
            # where W_b = V_{n+1}(q) - V_{n+1}(q+1)
            # So the added value is P_b(b + W_b) + P_a(a + W_a)
            
            val_b = P_b * (b_opt + W_b) if can_bid else 0.0
            val_a = P_a * (a_opt + W_a) if can_ask else 0.0
            
            V_curr[i_q] = V_next[i_q] - inv_cost[i_q] + val_b + val_a
            
        V_next = V_curr
        
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths.
    """
    # Identify regime key
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _CACHE:
        _CACHE[key] = _compute_optimal_depths(params)
    
    opt_bid, opt_ask = _CACHE[key]
    
    steps = params['steps']
    Q = params['Q']
    
    # Determine current step index
    # t = n * dt => n = t / dt = t * steps / T
    n = int(round(t * steps / params['T']))
    n = np.clip(n, 0, steps - 1)
    
    # Map inventory to index
    # inventory is in [-Q, Q]
    # idx = inventory + Q
    idx = inventory + Q
    
    # Get depths
    bid_depths = opt_bid[n, idx]
    ask_depths = opt_ask[n, idx]
    
    # Ensure shapes and types
    bid_depths = np.asarray(bid_depths, dtype=np.float64)
    ask_depths = np.asarray(ask_depths, dtype=np.float64)
    
    # Replace inf with a large number? No, np.inf is allowed by contract ("np.inf means no quote").
    # But contract says "Values below -3/k ... make result invalid". Inf is fine.
    # However, some systems might not handle Inf well in subsequent calculations if not expected.
    # The contract explicitly says "np.inf means no quote".
    
    return bid_depths, ask_depths
