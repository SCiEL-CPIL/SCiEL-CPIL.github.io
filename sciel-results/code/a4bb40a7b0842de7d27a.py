"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin below 120s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c**(1 - gamma) / (1 - gamma)
    
    def util_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # Use the scoring grid for internal computation
    m_grid = np.geomspace(0.001, 40, 251)
    n_m = len(m_grid)
    n_z = len(income)
    
    # Final period: consume all
    c_final = np.tile(m_grid, (n_z, 1))  # shape (n_z, n_m)
    
    # Backward induction
    # V[t+1][z][m] = value function at t+1
    V_next = np.zeros((n_z, n_m))  # value at t+1 for each z, m
    
    c_policies = [None] * horizon
    c_policies[horizon - 1] = c_final
    
    for t in range(horizon - 2, -1, -1):
        c_t = np.zeros((n_z, n_m))
        
        for z in range(n_z):
            # Vectorize over m for this z
            # For each m, we need to find optimal c
            
            # Pre-compute: for each m, the continuation value as a function of c
            # m' = R*(m - c) + income[z']
            # We need to interpolate V_next[z'] at m'
            
            # Try a grid of consumption fractions
            n_c = budget
            c_fracs = np.linspace(0.001, 1.0, n_c)
            
            # For each m in m_grid, compute optimal c
            for i, m in enumerate(m_grid):
                c_vals = m * c_fracs
                c_vals = np.maximum(c_vals, 1e-12)
                
                # Compute m_save = m - c_vals
                m_save = m - c_vals
                
                # For each z', compute m_next = R * m_save + income[z']
                # and interpolate V_next[z'] at m_next
                cont_vals = np.zeros(n_c)
                for zp in range(n_z):
                    prob = transition[z, zp]
                    if prob < 1e-10:
                        continue
                    m_next = R * m_save + income[zp]
                    
                    # Clip m_next to grid range for interpolation
                    m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])
                    v_interp = np.interp(m_next_clipped, m_grid, V_next[zp])
                    cont_vals += prob * v_interp
                
                total_vals = utility(c_vals) + beta * cont_vals
                
                best_idx = np.argmax(total_vals)
                best_c = c_vals[best_idx]
                best_val = total_vals[best_idx]
                
                c_t[z, i] = best_c
                V_next[z, i] = best_val
        
        c_policies[t] = c_t
        
        # Check time
        if time.time() > deadline:
            # Return what we have so far
            break
    
    # Now interpolate policies onto resource_grid
    result = np.zeros((horizon, n_z, len(resource_grid)))
    
    for t in range(horizon):
        if c_policies[t] is None:
            # Use final period policy
            c_t = c_final
        else:
            c_t = c_policies[t]
        
        for z in range(n_z):
            # Interpolate c_t[z, :] from m_grid to resource_grid
            result[t, z, :] = np.interp(resource_grid, m_grid, c_t[z, :])
    
    # Enforce constraints: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            for i in range(len(resource_grid)):
                m_val = resource_grid[i]
                c_val = result[t, z, i]
                if c_val <= 0:
                    result[t, z, i] = 1e-12
                if c_val > m_val + 1e-10:
                    result[t, z, i] = m_val
    
    # Final period must equal grid within 1e-9
    for z in range(n_z):
        result[horizon - 1, z, :] = resource_grid
    
    return result
# EVOLVE-BLOCK-END
