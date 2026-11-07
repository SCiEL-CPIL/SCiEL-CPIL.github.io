import numpy as np
import time

# Global cache for precomputed tables
_TABLE_CACHE = {}

def _get_tables(params):
    """
    Precomputes the optimal policy tables for the given parameters.
    Returns (OptB, OptA) arrays of shape (steps, 2*Q+1).
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
        
    dt = T / steps
    n_q = 2 * Q + 1
    q_indices = np.arange(-Q, Q + 1)
    
    # Grid for probabilities
    P_grid = 200
    # Avoid p=1 exactly to prevent log(0) issues in b calculation
    p_vals = np.linspace(0, 1 - 1e-15, P_grid)
    
    # Calculate depths b and a from p
    # p = 1 - exp(-A exp(-k b) dt)
    # exp(-k b) = -ln(1-p) / (A dt)
    # -k b = ln(-ln(1-p) / (A dt))
    # b = -1/k * ln(-ln(1-p) / (A dt))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        b_vals = -1.0 / k * np.log(-np.log(1 - p_vals) / (A * dt))
        a_vals = b_vals.copy() # Symmetric
        
    # Mask invalid depths (b < -3/k)
    min_depth = -3.0 / k
    valid_b = b_vals >= min_depth
    valid_a = a_vals >= min_depth
    
    # If a depth is invalid, the probability is not p_vals, but we just mark it as -inf objective
    # We will create 2D grids
    p_b_grid, p_a_grid = np.meshgrid(p_vals, p_vals, indexing='ij')
    b_grid, a_grid = np.meshgrid(b_vals, a_vals, indexing='ij')
    
    # Invalid mask: if b < min_depth OR a < min_depth
    invalid_mask = (~valid_b)[:, None] | (~valid_a)[None, :]
    
    # Precompute V_next
    V_next = np.zeros(n_q)
    
    OptB = np.full((steps, n_q), np.inf)
    OptA = np.full((steps, n_q), np.inf)
    
    # Precompute inventory cost coefficients
    inv_cost_coeff = 0.5 * gamma * sigma**2 * dt
    
    # Loop backwards in time
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_q)
        
        # V_{q-1}, V_q, V_{q+1}
        # For q=-Q, V_{q-1} is not used (p_b=0)
        # For q=Q, V_{q+1} is not used (p_a=0)
        
        V_q = V_next
        V_qm1 = np.roll(V_next, 1)
        V_qm1[0] = 0 # Dummy, will be masked
        V_qp1 = np.roll(V_next, -1)
        V_qp1[-1] = 0 # Dummy, will be masked
        
        # Expand to (n_q, P, P)
        # V terms
        Vq_t = V_q[:, None, None]
        Vqm1_t = V_qm1[:, None, None]
        Vqp1_t = V_qp1[:, None, None]
        
        # Grid terms (1, P, P)
        pb_t = p_b_grid[None, :, :]
        pa_t = p_a_grid[None, :, :]
        b_t = b_grid[None, :, :]
        a_t = a_grid[None, :, :]
        
        # Objective J
        # J = -inv_cost*q^2 + b*p_b + a*p_a + V_{q-1} p_b (1-p_a) + V_{q+1} p_a (1-p_b) + V_q (1 - p_b - p_a + 2 p_b p_a)
        
        q_sq = (q_indices**2)[:, None, None]
        cost_term = -inv_cost_coeff * q_sq
        
        reward_b = b_t * pb_t
        reward_a = a_t * pa_t
        
        term_b = Vqm1_t * pb_t * (1 - pa_t)
        term_a = Vqp1_t * pa_t * (1 - pb_t)
        term_v = Vq_t * (1 - pb_t - pa_t + 2 * pb_t * pa_t)
        
        J = cost_term + reward_b + reward_a + term_b + term_a + term_v
        
        # Apply invalid mask
        J[invalid_mask[None, :, :]] = -np.inf
        
        # Apply boundary conditions
        # q = -Q (index 0): p_b must be 0. Only p_b=0 is valid?
        # Our grid has p_b=0 at index 0.
        # So for q=-Q, we want to select only the column where p_b=0.
        # We can set J[0, 1:, :] = -inf
        J[0, 1:, :] = -np.inf
        
        # q = Q (index -1): p_a must be 0.
        # J[-1, :, 1:] = -inf
        J[-1, :, 1:] = -np.inf
        
        # Find max
        max_idx = np.argmax(J, axis=(1, 2))
        flat_idx = np.argmax(J.reshape(n_q, -1), axis=1)
        row_idx, col_idx = np.unravel_index(flat_idx, (P_grid, P_grid))
        
        # Update V_curr
        V_curr = np.max(J, axis=(1, 2))
        
        # Store optimal depths
        OptB[n, :] = b_vals[row_idx]
        OptA[n, :] = a_vals[col_idx]
        
        V_next = V_curr
        
    _TABLE_CACHE[key] = (OptB, OptA)
    return OptB, OptA

def quote(t, inventory, mid, params):
    """
    Returns (bid_depths, ask_depths) for the given state.
    """
    # Get or compute tables
    OptB, OptA = _get_tables(params)
    
    steps = params['steps']
    T = params['T']
    dt = T / steps
    
    # Determine step index
    n = int(t / dt)
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    Q = params['Q']
    
    # Map inventory to index
    # inventory is in [-Q, Q]
    # index = inventory + Q
    inv_idx = inventory + Q
    
    # Clip indices to valid range just in case
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Retrieve depths
    bid_depths = OptB[n, inv_idx]
    ask_depths = OptA[n, inv_idx]
    
    return bid_depths, ask_depths
