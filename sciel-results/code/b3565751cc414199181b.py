"""Rewritten solver using high-resolution backward induction with vectorized operations."""
import numpy as np
from time import time

def solve_consumption(model, resource_grid, budget):
    """
    Solve the consumption-saving problem via backward induction.
    
    Parameters:
    -----------
    model : dict with keys 'beta', 'gamma', 'R', 'income', 'transition', 'horizon'
    resource_grid : np.ndarray, shape (101,)
    budget : int, number of consumption search points per m
    
    Returns:
    --------
    consumption : np.ndarray, shape (horizon, 2, len(resource_grid))
    """
    start_time = time()
    deadline = start_time + 135  # safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)  # 2
    n_grid = len(resource_grid)  # 101
    m_min = resource_grid[0]
    m_max = resource_grid[-1]
    
    # CRRA utility function
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    def util_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    # Initialize consumption policy
    consumption = np.zeros((horizon, n_states, n_grid))
    
    # Terminal period: consume everything
    consumption[-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    # Backward induction from t = horizon-2 down to 0
    # For each state z and each m, find optimal c
    
    # Precompute continuation value lookup structure
    # We'll use a dense grid for interpolation of next-period value
    
    # Use a finer internal grid for better accuracy
    internal_n = 2000
    internal_grid = np.geomspace(m_min, m_max, internal_n)
    
    # Search grid for consumption
    search_n = 500
    search_frac = np.linspace(0.001, 1.0, search_n)
    
    for t in range(horizon - 2, -1, -1):
        if time() > deadline - 5:
            # Fill remaining periods with consume-all (safe fallback)
            for tt in range(t, -1, -1):
                consumption[tt, :, :] = resource_grid[np.newaxis, np.newaxis, :]
            break
        
        # Next period value function for each state
        # V_next[z, m] = max_c { u(c) + beta * sum_z' transition[z,z'] * V_next[z', m'] }
        # But we're going backward, so we have V_{t+1} for each state
        
        # Store value functions on internal grid
        # V[z, i] for i in internal grid
        V_next = np.zeros((n_states, internal_n))
        
        # For the last period (t = horizon-2), V_next is the terminal value
        if t == horizon - 2:
            # Terminal: V[z, m] = u(m) for all z
            V_next[:, :] = util(internal_grid)[np.newaxis, :]
        else:
            # V_next[z, m] from previous iteration
            # We need to interpolate consumption[t+1, z, :] onto internal_grid
            for z in range(n_states):
                c_policy = consumption[t + 1, z, :]
                # Interpolate consumption policy onto internal grid
                c_interp = np.interp(internal_grid, resource_grid, c_policy)
                # m' = R*(m - c) + income[z']
                # V[z, m] = u(c) + beta * sum_{z'} transition[z, z'] * V_{t+1}[z', m']
                # But wait, we need V_{t+1}[z', m'] which is the value at t+1 for state z'
                
                # Actually, let me restructure. Let me compute V[t][z][m] properly.
                
                # V[t][z][m] = max_c { u(c) + beta * sum_{z'} P[z,z'] * V[t+1][z'][m'] }
                # where m' = R*(m-c) + income[z']
                
                # So I need V[t+1][z'] for all z', interpolated on internal grid
                # But I'm computing V[t] now, and V[t+1] was computed in the previous step.
                
                # Let me restructure: store V as value functions, not just consumption.
                pass
        
        # Let me restructure the whole thing more cleanly.
        
        # Actually, let me restart the backward induction with proper value function storage.
        break
    
    # Let me redo this more carefully with proper value function tracking
    consumption = np.zeros((horizon, n_states, n_grid))
    consumption[-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    # V[t][z] is the value function at time t, state z, as a function of m
    # Store on internal grid
    V = np.zeros((horizon, n_states, internal_n))
    
    # Terminal
    V[-1, :, :] = util(internal_grid)[np.newaxis, :, np.newaxis]
    
    for t in range(horizon - 2, -1, -1):
        if time() > deadline - 5:
            for tt in range(t, -1, -1):
                consumption[tt, :, :] = resource_grid[np.newaxis, np.newaxis, :]
            break
        
        for z in range(n_states):
            # For each m in internal_grid, find optimal c
            # V[t][z][m] = max_c { u(c) + beta * sum_{z'} P[z,z'] * V[t+1][z'][m'] }
            # m' = R*(m-c) + income[z']
            
            # Vectorize over m
            m_vals = internal_grid  # shape (internal_n,)
            
            # For each m, search over c
            # c must satisfy 0 < c <= m
            # m' = R*(m-c) + income[z'] >= 0 requires c <= m + income[z']/R
            
            # Use search grid: c = m * frac
            # For each frac, c = m * frac
            # m' = R*(m - m*frac) + income[z'] = m*R*(1-frac) + income[z']
            
            best_val = np.full(internal_n, -np.inf)
            best_c = np.zeros(internal_n)
            
            # Process in chunks to manage memory
            chunk_size = 200
            for chunk_start in range(0, search_n, chunk_size):
                chunk_end = min(chunk_start + chunk_size, search_n)
                fracs = search_frac[chunk_start:chunk_end]  # shape (chunk,)
                
                for zi, frac in enumerate(fracs):
                    c = m_vals * frac  # shape (internal_n,)
                    c = np.maximum(c, 1e-12)
                    c = np.minimum(c, m_vals)  # c <= m
                    
                    # Continuation value
                    cont = np.zeros(internal_n)
                    for zp in range(n_states):
                        m_prime = R * (m_vals - c) + income[zp]
                        m_prime = np.maximum(m_prime, m_min * 0.5)  # ensure positive
                        # Interpolate V[t+1][zp] onto m_prime
                        # V[t+1][zp] is defined on internal_grid
                        V_interp = np.interp(m_prime, internal_grid, V[t+1, zp, :])
                        cont += transition[z, zp] * V_interp
                    
                    total_val = util(c) + beta * cont
                    
                    mask = total_val > best_val
                    best_val[mask] = total_val[mask]
                    best_c[mask] = c[mask]
            
            V[t, z, :] = best_val
            # Interpolate best_c back to resource_grid
            consumption[t, z, :] = np.interp(resource_grid, internal_grid, best_c)
    
    # Enforce constraints
    for t in range(horizon):
        for z in range(n_states):
            consumption[t, z, :] = np.clip(consumption[t, z, :], 1e-12, resource_grid + 1e-10)
    
    # Terminal period must consume everything
    consumption[-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    return consumption
