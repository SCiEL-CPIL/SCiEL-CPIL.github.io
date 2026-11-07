import numpy as np
from functools import lru_cache

# We need to handle the fact that params are a dict.
# We will create a helper to precompute.

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
    
    # Initialize value function
    V_next = np.zeros(num_states)
    
    # Tables for optimal depths
    opt_bid = np.zeros((N, num_states))
    opt_ask = np.zeros((N, num_states))
    
    # Grid for depth search
    # Max depth where probability is non-negligible
    # p = 1 - exp(-A * exp(-k*d) * dt)
    # If A * exp(-k*d) * dt < 1e-5, p is very small.
    # exp(-k*d) < 1e-5 / (A*dt)
    # -k*d < log(1e-5 / (A*dt))
    # d > log(A*dt / 1e-5) / k
    d_max = np.log(max(A * dt / 1e-5, 1.0)) / k
    d_max = np.clip(d_max, 1.0, 20.0) # Clamp to reasonable range
    
    num_grid = 100
    d_grid = np.linspace(0, d_max, num_grid)
    d_b_grid, d_a_grid = np.meshgrid(d_grid, d_grid, indexing='ij')
    d_b_flat = d_b_grid.flatten()
    d_a_flat = d_a_grid.flatten()
    
    # Precompute probabilities for the grid
    p_b_grid = 1 - np.exp(-A * np.exp(-k * d_b_flat) * dt)
    p_a_grid = 1 - np.exp(-A * np.exp(-k * d_a_flat) * dt)
    
    # Coefficients for the linear combination of V_next terms
    # Term: p_b(1-p_a) * Delta_plus + (1-p_b)p_a * Delta_minus + (1 - p_b - p_a + 2*p_b*p_a) * V_curr
    C_grid = p_b_grid * (1 - p_a_grid)
    D_grid = (1 - p_b_grid) * p_a_grid
    E_grid = 1 - p_b_grid - p_a_grid + 2 * p_b_grid * p_a_grid
    
    # Immediate reward coefficients (linear in d_b, d_a)
    # Reward = p_b * d_b + p_a * d_a
    # This is not linear in the grid indices, but we have the values.
    R_grid = p_b_grid * d_b_flat + p_a_grid * d_a_flat
    
    # For each step from N-1 down to 0
    for n in range(N - 1, -1, -1):
        V_curr = np.zeros(num_states)
        
        # Calculate deltas for all q
        # q index 0 corresponds to -Q, index num_states-1 corresponds to Q
        # Delta_plus[q] = V_next[q+1] - V_next[q]
        # Valid for q < Q (index < num_states - 1)
        Delta_plus = np.zeros(num_states)
        Delta_minus = np.zeros(num_states)
        
        # q from -Q to Q-1 (indices 0 to num_states-2)
        Delta_plus[:-1] = V_next[1:] - V_next[:-1]
        # q from -Q+1 to Q (indices 1 to num_states-1)
        Delta_minus[1:] = V_next[:-1] - V_next[1:]
        
        # Construct the value matrix for all grid points and all states
        # Shape: (num_grid_points, num_states)
        # Val = R_grid[:, None] + C_grid[:, None] * Delta_plus[None, :] + D_grid[:, None] * Delta_minus[None, :] + E_grid[:, None] * V_next[None, :]
        
        # We need to handle boundaries where fills are impossible
        # If q = Q (last index), no bid fill. p_b should be 0.
        # If q = -Q (first index), no ask fill. p_a should be 0.
        
        # We can just compute the max and then override the depths for boundary states.
        # However, the value calculation must be correct.
        # For q=Q, the term with Delta_plus is multiplied by p_b(1-p_a).
        # If we force p_b=0, the term vanishes.
        # In the grid, p_b is never exactly 0 unless d_b is huge.
        # But we are maximizing over d_b. If q=Q, the optimal strategy is to not bid (d_b=inf).
        # The value function calculation should reflect that.
        # If we allow d_b to be small, we might "cheat" by getting a fill that is actually forbidden.
        # So we must constrain the optimization for boundary states.
        
        # Strategy:
        # 1. Compute max value for all states assuming all fills possible.
        # 2. For q=Q, recompute max value with p_b=0 (i.e., only vary d_a).
        # 3. For q=-Q, recompute max value with p_a=0 (i.e., only vary d_b).
        
        # This is inefficient. Let's just mask the grid or handle it separately.
        # Given the small size, we can just do a loop over states if needed, but vectorization is better.
        
        # Let's compute the full matrix first.
        Val_matrix = R_grid[:, None] + C_grid[:, None] * Delta_plus[None, :] + D_grid[:, None] * Delta_minus[None, :] + E_grid[:, None] * V_next[None, :]
        
        # Subtract cost
        # Cost = gamma/2 * sigma^2 * q^2 * dt
        q_vals = np.arange(-Q, Q + 1)
        costs = 0.5 * gamma * sigma**2 * q_vals**2 * dt
        Val_matrix -= costs[None, :]
        
        # Find best grid point for each state
        best_idx = np.argmax(Val_matrix, axis=0)
        V_curr = Val_matrix[best_idx, np.arange(num_states)]
        
        # Get optimal depths
        opt_d_b = d_b_flat[best_idx]
        opt_d_a = d_a_flat[best_idx]
        
        # Handle boundaries
        # q = Q is index num_states - 1
        # q = -Q is index 0
        
        # For q = Q, no bid. Set d_b = inf.
        # The value should be computed with p_b=0.
        # Let's recompute the value for q=Q and q=-Q properly.
        
        if num_states > 1:
            # q = Q (index -1)
            q_idx = num_states - 1
            # Only ask side matters. p_b = 0.
            # Maximize p_a * d_a + (1-p_a) * Delta_minus[q_idx] + p_a * V_next[q_idx] (wait, E term?)
            # If p_b=0:
            # Prob(q+1) = 0
            # Prob(q-1) = p_a
            # Prob(q) = 1 - p_a
            # Value = p_a * d_a + p_a * V_next[q-1] + (1-p_a) * V_next[q] - cost
            #      = p_a * (d_a + V_next[q-1] - V_next[q]) + V_next[q] - cost
            #      = p_a * (d_a + Delta_minus[q]) + V_next[q] - cost
            
            # We need to maximize over d_a.
            # p_a = 1 - exp(-A exp(-k d_a) dt)
            # Let's just search over the d_a grid.
            p_a_vals = 1 - np.exp(-A * np.exp(-k * d_grid) * dt)
            vals_qQ = p_a_vals * (d_grid + Delta_minus[q_idx]) + V_next[q_idx] - costs[q_idx]
            best_a_idx = np.argmax(vals_qQ)
            opt_d_a[q_idx] = d_grid[best_a_idx]
            opt_d_b[q_idx] = np.inf
            V_curr[q_idx] = vals_qQ[best_a_idx]
            
            # q = -Q (index 0)
            q_idx = 0
            # Only bid side matters. p_a = 0.
            # Prob(q+1) = p_b
            # Prob(q-1) = 0
            # Prob(q) = 1 - p_b
            # Value = p_b * d_b + p_b * V_next[q+1] + (1-p_b) * V_next[q] - cost
            #      = p_b * (d_b + V_next[q+1] - V_next[q]) + V_next[q] - cost
            #      = p_b * (d_b + Delta_plus[q]) + V_next[q] - cost
            
            p_b_vals = 1 - np.exp(-A * np.exp(-k * d_grid) * dt)
            vals_qmQ = p_b_vals * (d_grid + Delta_plus[q_idx]) + V_next[q_idx] - costs[q_idx]
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
    # Key for cache
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _cache:
        _cache[key] = _precompute_tables(params)
    
    opt_bid, opt_ask = _cache[key]
    
    steps = params['steps']
    T = params['T']
    dt = T / steps
    Q = params['Q']
    
    n = int(t / dt)
    # Clamp n to [0, steps-1]
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    # Map inventory to index
    # inventory is in [-Q, Q]
    # index = inventory + Q
    idx = inventory + Q
    
    # Ensure idx is within bounds
    idx = np.clip(idx, 0, 2 * Q)
    
    bid_depths = opt_bid[n, idx]
    ask_depths = opt_ask[n, idx]
    
    return bid_depths, ask_depths
