import numpy as np
from scipy.optimize import brentq
import time

# Cache for precomputed tables
_TABLE_CACHE = {}

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute optimal bid and ask depths for all steps and inventories.
    Uses vectorized grid search for speed.
    """
    dt = T / steps
    N = steps
    max_q = Q
    min_q = -Q
    n_states = 2 * Q + 1
    
    V_next = np.zeros(n_states)
    opt_bid = np.zeros((N, n_states))
    opt_ask = np.zeros((N, n_states))
    
    # Create a grid of candidate depths for vectorized evaluation
    # The optimal depth is typically in [0, ~5/k]. Let's use a fine grid.
    # We need enough resolution to be accurate.
    n_grid = 500
    grid = np.linspace(0, 5.0 / k, n_grid)
    
    for n in range(N - 1, -1, -1):
        V_curr = np.zeros(n_states)
        
        # For each inventory state, compute the optimal bid and ask depth
        # We'll vectorize over inventory states
        
        # Get deltas
        delta_plus = np.zeros(n_states)
        delta_minus = np.zeros(n_states)
        
        # q = q_idx - Q
        q_vals = np.arange(n_states) - Q
        
        # delta_plus[i] = V_next[i+1] - V_next[i] for q < max_q
        valid_plus = q_vals < max_q
        if np.any(valid_plus):
            delta_plus[valid_plus] = V_next[1:n_states] - V_next[0:n_states-1]
            # Only for valid indices
            delta_plus[valid_plus] = V_next[q_vals[valid_plus] + 1] - V_next[q_vals[valid_plus]]
        
        # delta_minus[i] = V_next[i-1] - V_next[i] for q > min_q
        valid_minus = q_vals > min_q
        if np.any(valid_minus):
            delta_minus[valid_minus] = V_next[q_vals[valid_minus] - 1] - V_next[q_vals[valid_minus]]
        
        # Now solve for optimal b and a iteratively
        # Initialize
        b_vals = np.zeros(n_states)
        a_vals = np.zeros(n_states)
        
        for _iter in range(15):
            # Solve for b given a
            # C_b = delta_plus * (1 - p_a)
            p_a_curr = 1.0 - np.exp(-A * np.exp(-k * a_vals) * dt)
            C_b = delta_plus * (1.0 - p_a_curr)
            
            # For states where q == max_q, force b = 0
            # For others, find optimal b from grid
            
            # Evaluate J_b(d) = p(d) * (d + C_b) for all d in grid, for all states
            # p(d) = 1 - exp(-A * exp(-k*d) * dt)
            # We want argmax over d for each state
            
            # Shape: (n_grid, n_states)
            p_grid = 1.0 - np.exp(-A * np.exp(-k * grid[:, None]) * dt)  # (n_grid, 1) broadcast
            # J_b[i, j] = p_grid[i] * (grid[i] + C_b[j])
            # But C_b varies by state j
            J_b = p_grid * (grid[:, None] + C_b[None, :])  # (n_grid, n_states)
            
            # For q == max_q, set J_b to -inf except at d=0
            if np.any(~valid_plus):
                J_b[:, ~valid_plus] = -np.inf
                J_b[0, ~valid_plus] = 0.0  # d=0 is valid
            
            b_new = grid[np.argmax(J_b, axis=0)]
            
            # Solve for a given b
            p_b_curr = 1.0 - np.exp(-A * np.exp(-k * b_new) * dt)
            C_a = delta_minus * (1.0 - p_b_curr)
            
            # Evaluate J_a(d) = p(d) * (d + C_a)
            J_a = p_grid * (grid[:, None] + C_a[None, :])  # (n_grid, n_states)
            
            # For q == min_q, set J_a to -inf except at d=0
            if np.any(~valid_minus):
                J_a[:, ~valid_minus] = -np.inf
                J_a[0, ~valid_minus] = 0.0
            
            a_new = grid[np.argmax(J_a, axis=0)]
            
            # Check convergence
            if np.max(np.abs(b_new - b_vals)) < 1e-6 and np.max(np.abs(a_new - a_vals)) < 1e-6:
                b_vals = b_new
                a_vals = a_new
                break
            b_vals = b_new
            a_vals = a_new
        
        opt_bid[n, :] = b_vals
        opt_ask[n, :] = a_vals
        
        # Calculate V_curr
        p_b = 1.0 - np.exp(-A * np.exp(-k * b_vals) * dt)
        p_a = 1.0 - np.exp(-A * np.exp(-k * a_vals) * dt)
        
        # For boundary states, adjust probabilities
        # q == max_q: no bid fill
        if np.any(~valid_plus):
            p_b[~valid_plus] = 0.0
        
        # q == min_q: no ask fill
        if np.any(~valid_minus):
            p_a[~valid_minus] = 0.0
        
        # Transitions
        prob_q_plus = p_b * (1.0 - p_a)
        prob_q_minus = (1.0 - p_b) * p_a
        prob_q = 1.0 - prob_q_plus - prob_q_minus
        
        reward = p_b * b_vals + p_a * a_vals
        
        # Expected value from next step
        # V_next[q+1] for q < max_q, V_next[q-1] for q > min_q, V_next[q] otherwise
        V_q_plus = np.zeros(n_states)
        V_q_minus = np.zeros(n_states)
        V_q_same = V_next.copy()
        
        if np.any(valid_plus):
            V_q_plus[valid_plus] = V_next[q_vals[valid_plus] + 1]
        if np.any(valid_minus):
            V_q_minus[valid_minus] = V_next[q_vals[valid_minus] - 1]
        
        exp_val = prob_q_plus * V_q_plus + prob_q_minus * V_q_minus + prob_q * V_q_same
        
        risk = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
        
        V_curr = reward - risk + exp_val
        V_next = V_curr
    
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    # Determine step index
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    # Check cache
    key = (params['sigma'], params['A'], params['k'], params['gamma'])
    if key not in _TABLE_CACHE:
        # Precompute
        start_time = time.time()
        opt_bid, opt_ask = _precompute_regime(
            params['sigma'], params['A'], params['k'], params['gamma'], 
            params['T'], params['steps'], params['Q']
        )
        _TABLE_CACHE[key] = (opt_bid, opt_ask)
        
        # Time guard (safety)
        if time.time() - start_time > 300:
             depth = 1.0 / params["k"]
             return np.full(len(inventory), depth), np.full(len(inventory), depth)

    opt_bid, opt_ask = _TABLE_CACHE[key]
    
    # Map inventory to indices
    q_indices = np.array(inventory) + params['Q']
    
    # Look up depths
    bid_depths = opt_bid[n, q_indices]
    ask_depths = opt_ask[n, q_indices]
    
    return bid_depths, ask_depths
