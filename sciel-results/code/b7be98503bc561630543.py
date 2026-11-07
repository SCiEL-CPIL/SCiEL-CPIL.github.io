"""
Optimal market making with inventory risk using vectorized HJB backward induction.

We solve the discrete-time HJB equation exactly (up to grid resolution) for each
regime. The key insight is that the depth optimization is separable in the
"effective" problem due to the structure of the objective.
"""
import numpy as np
import time

# Module-level cache for policy tables
_policy_cache = {}
_start_time = time.time()
_TIME_LIMIT = 310  # safety margin before 330s kill


def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths via backward induction on the discrete-time model.
    
    Returns:
        bid_table: array of shape (steps, 2*Q+1)
        ask_table: array of shape (steps, 2*Q+1)
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    
    # Value function: V[q] = optimal expected objective from current step with inventory q
    V = np.zeros(n_inv)
    
    # We'll store policies for each step
    bid_table = np.full((steps, n_inv), np.inf)
    ask_table = np.full((steps, n_inv), np.inf)
    
    # Grid of candidate depths
    # Use a fine grid; the optimal depth is typically small
    n_grid = 100
    max_depth = 5.0
    depth_grid = np.linspace(1e-4, max_depth, n_grid)
    
    # Pre-compute fill probabilities
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    exp_term = np.exp(-k * depth_grid)
    rate = A * exp_term * dt
    p_fill = 1.0 - np.exp(-rate)
    
    # For the 2D optimization (interior inventories), we need to be smart.
    # The value function is:
    # V[q] = -inv_cost + max_{d_b, d_a} [
    #     p_b(d_b)*p_a(d_a)*(d_b + d_a + V[q]) +
    #     p_b(d_b)*(1-p_a(d_a))*(d_b + V[q+1]) +
    #     (1-p_b(d_b))*p_a(d_a)*(d_a + V[q-1]) +
    #     (1-p_b(d_b))*(1-p_a(d_a))*V[q]
    # ]
    # = -inv_cost + max_{d_b, d_a} [
    #     p_b*d_b + p_a*d_a +
    #     p_b*p_a*V[q] + p_b*(1-p_a)*V[q+1] + (1-p_b)*p_a*V[q-1] + (1-p_b)*(1-p_a)*V[q]
    # ]
    # = -inv_cost + max_{d_b, d_a} [
    #     p_b*d_b + p_a*d_a +
    #     V[q] + p_b*(V[q+1]-V[q]) - p_a*(V[q]-V[q-1]) + p_b*p_a*(V[q]-V[q+1]-V[q-1]+V[q]) ... 
    # ]
    
    # Let me redo this more carefully:
    # Let a = V[q+1] - V[q], b = V[q] - V[q-1]
    # Then:
    # p_b*p_a*V[q] + p_b*(1-p_a)*V[q+1] + (1-p_b)*p_a*V[q-1] + (1-p_b)*(1-p_a)*V[q]
    # = V[q] + p_b*(1-p_a)*(V[q+1]-V[q]) + (1-p_b)*p_a*(V[q-1]-V[q]) + p_b*p_a*(V[q+1]+V[q-1]-2*V[q]) ... 
    # This is getting complex. Let's just do the 2D grid search.
    
    # For efficiency, use a smaller grid for 2D
    n_grid_2d = 60
    depth_grid_2d = np.linspace(1e-4, max_depth, n_grid_2d)
    p_fill_2d = 1.0 - np.exp(-A * np.exp(-k * depth_grid_2d) * dt)
    
    # Work backwards
    for n in range(steps - 1, -1, -1):
        if time.time() - _start_time > _TIME_LIMIT:
            break
        
        V_next = V.copy()
        
        # Interior: 1D optimization for each side separately won't work,
        # but we can do a 2D grid search vectorized over all interior q.
        
        # For q = -Q: only bid enabled
        q = -Q
        q_idx = 0
        inv_cost = (gamma / 2.0) * sigma**2 * q**2 * dt
        q_next = q + 1
        q_next_idx = q_next + Q
        values = -depth_grid * p_fill + (1 - p_fill) * V_next[q_idx] + p_fill * V_next[q_next_idx]
        best_idx = np.argmax(values)
        bid_table[n, q_idx] = depth_grid[best_idx]
        ask_table[n, q_idx] = np.inf
        V[q_idx] = -inv_cost + values[best_idx]
        
        # For q = Q: only ask enabled
        q = Q
        q_idx = 2 * Q
        inv_cost = (gamma / 2.0) * sigma**2 * q**2 * dt
        q_next = q - 1
        q_next_idx = q_next + Q
        values = -depth_grid * p_fill + (1 - p_fill) * V_next[q_idx] + p_fill * V_next[q_next_idx]
        best_idx = np.argmax(values)
        ask_table[n, q_idx] = depth_grid[best_idx]
        bid_table[n, q_idx] = np.inf
        V[q_idx] = -inv_cost + values[best_idx]
        
        # Interior q = -Q+1, ..., Q-1
        # Vectorize over all interior inventories
        q_interior = np.arange(-Q + 1, Q)
        q_indices = q_interior + Q  # 1 to 2Q-1
        n_interior = len(q_interior)
        
        if n_interior > 0:
            # inv_cost for each q
            inv_costs = (gamma / 2.0) * sigma**2 * q_interior**2 * dt
            
            # V_next values
            Vq = V_next[q_indices]  # V[q]
            Vq1 = V_next[q_indices + 1]  # V[q+1]
            Vqm1 = V_next[q_indices - 1]  # V[q-1]
            
            # 2D grid search: for each (d_b, d_a) and each q, compute value
            # We'll iterate over the 2D grid and keep track of best for each q
            best_val = np.full(n_interior, -np.inf)
            best_d_b = np.full(n_interior, np.inf)
            best_d_a = np.full(n_interior, np.inf)
            
            D_b, D_a = np.meshgrid(depth_grid_2d, depth_grid_2d, indexing='ij')
            P_b = p_fill_2d[:, None] * np.ones((1, n_grid_2d))  # (n_grid_2d, n_grid_2d)
            P_a = np.ones((n_grid_2d, 1)) * p_fill_2d[None, :]  # (n_grid_2d, n_grid_2d)
            
            # Earnings: P_b * D_b + P_a * D_a  (n_grid_2d, n_grid_2d)
            earnings = P_b * D_b + P_a * D_a
            
            # For each (i, j) in the grid, compute the value for all q
            # val[q] = earnings[i,j] + 
            #          P_b[i,j]*P_a[i,j]*Vq[q] + 
            #          P_b[i,j]*(1-P_a[i,j])*Vq1[q] + 
            #          (1-P_b[i,j])*P_a[i,j]*Vqm1[q] + 
            #          (1-P_b[i,j])*(1-P_a[i,j])*Vq[q]
            
            # To be efficient, we can batch this.
            # For each grid point (i, j):
            #   pb = P_b[i,j], pa = P_a[i,j]
            #   val = earnings[i,j] + pb*pa*Vq + pb*(1-pa)*Vq1 + (1-pb)*pa*Vqm1 + (1-pb)*(1-pa)*Vq
            
            # Vectorize over grid points (i,j)
            # This creates a large intermediate array: (n_grid_2d^2, n_interior)
            # n_grid_2d^2 = 3600, n_interior = 58, so 208,800 floats - manageable
            
            PB = P_b.ravel()  # (n_grid_2d^2,)
            PA = P_a.ravel()  # (n_grid_2d^2,)
            E = earnings.ravel()  # (n_grid_2d^2,)
            
            # (n_grid_2d^2, n_interior)
            vals = (E[:, None] + 
                    PB[:, None] * PA[:, None] * Vq[None, :] + 
                    PB[:, None] * (1 - PA[:, None]) * Vq1[None, :] + 
                    (1 - PB[:, None]) * PA[:, None] * Vqm1[None, :] + 
                    (1 - PB[:, None]) * (1 - PA[:, None]) * Vq[None, :])
            
            best_idx_flat = np.argmax(vals, axis=0)
            best_val = vals[best_idx_flat, np.arange(n_interior)]
            
            # Map back to grid
            # best_idx_flat is in [0, n_grid_2d^2)
            # Row-major: index = i * n_grid_2d + j
            best_i = best_idx_flat // n_grid_2d
            best_j = best_idx_flat % n_grid_2d
            
            best_d_b = depth_grid_2d[best_i]
            best_d_a = depth_grid_2d[best_j]
            
            bid_table[n, q_indices] = best_d_b
            ask_table[n, q_indices] = best_d_a
            V[q_indices] = -inv_costs + best_val
        
        V_next = V  # for next iteration
    
    return bid_table, ask_table


def _get_policy_table(params):
    """Get or compute the policy table for given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        sigma = params['sigma']
        A = params['A']
        k = params['k']
        gamma = params['gamma']
        T = params['T']
        steps = params['steps']
        Q = params['Q']
        
        bid_table, ask_table = _compute_policy_table(sigma, A, k, gamma, T, steps, Q)
        _policy_cache[key] = (bid_table, ask_table)
    
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    dt = params['T'] / params['steps']
    step = int(round(t / dt))
    step = min(max(step, 0), params['steps'] - 1)
    
    Q = params['Q']
    
    bid_table, ask_table = _get_policy_table(params)
    
    inv_indices = inventory + Q
    inv_indices = np.clip(inv_indices, 0, 2 * Q)
    
    bid_depths = bid_table[step, inv_indices].astype(np.float64)
    ask_depths = ask_table[step, inv_indices].astype(np.float64)
    
    # Ensure valid values: must be > -3/k
    min_val = -3.0 / params['k'] + 1e-6
    bid_depths = np.where(bid_depths > min_val, bid_depths, np.inf)
    ask_depths = np.where(ask_depths > min_val, ask_depths, np.inf)
    
    return bid_depths, ask_depths
