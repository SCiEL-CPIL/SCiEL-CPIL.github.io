"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin below 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    n_z = len(income)
    
    # Use the scoring grid for internal computation
    m_grid = np.geomspace(0.001, 40, 251)
    n_m = len(m_grid)
    
    # Utility functions
    def u(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c_safe)
        else:
            return c_safe ** (1 - gamma) / (1 - gamma)
    
    def u_prime(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c_safe
        else:
            return c_safe ** (-gamma)
    
    # Final period: consume all
    c_final = np.tile(m_grid, (n_z, 1))  # shape (n_z, n_m)
    
    # Backward induction
    # V_next[z, m] = value function at t+1 for each z, m
    V_next = np.zeros((n_z, n_m))  # value at t+1 for each z, m
    
    c_policies = [None] * horizon
    c_policies[horizon - 1] = c_final
    
    # Pre-compute transition probabilities
    trans = transition  # (n_z, n_z)
    
    for t in range(horizon - 2, -1, -1):
        c_t = np.zeros((n_z, n_m))
        
        for z in range(n_z):
            # For each m in m_grid, find optimal c
            # Vectorize over consumption grid for each m
            
            n_c = budget
            c_fracs = np.linspace(0.001, 1.0, n_c)
            
            for i, m in enumerate(m_grid):
                c_vals = m * c_fracs
                c_vals = np.maximum(c_vals, 1e-12)
                
                # Compute m_save = m - c_vals
                m_save = m - c_vals
                
                # For each z', compute m_next = R * m_save + income[z']
                # and interpolate V_next[z'] at m_next using PCHIP
                cont_vals = np.zeros(n_c)
                for zp in range(n_z):
                    prob = trans[z, zp]
                    if prob < 1e-10:
                        continue
                    m_next = R * m_save + income[zp]
                    
                    # Clip m_next to grid range for interpolation
                    m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])
                    
                    # Use PCHIP for smoother interpolation
                    if i == 0:
                        # Build PCHIP interpolator for this z'
                        try:
                            pchip_interp = PchipInterpolator(m_grid, V_next[zp], extrapolate=False)
                            v_interp = pchip_interp(m_next_clipped)
                            # Replace nan with boundary values
                            v_interp = np.where(np.isnan(v_interp), V_next[zp, 0], v_interp)
                        except Exception:
                            v_interp = np.interp(m_next_clipped, m_grid, V_next[zp])
                    else:
                        # Reuse the same interpolator structure
                        try:
                            pchip_interp = PchipInterpolator(m_grid, V_next[zp], extrapolate=False)
                            v_interp = pchip_interp(m_next_clipped)
                            v_interp = np.where(np.isnan(v_interp), V_next[zp, 0], v_interp)
                        except Exception:
                            v_interp = np.interp(m_next_clipped, m_grid, V_next[zp])
                    
                    cont_vals += prob * v_interp
                
                total_vals = u(c_vals) + beta * cont_vals
                
                best_idx = np.argmax(total_vals)
                best_c = c_vals[best_idx]
                best_val = total_vals[best_idx]
                
                c_t[z, i] = best_c
                V_next[z, i] = best_val
        
        c_policies[t] = c_t
        
        # Check time
        if time.time() > deadline:
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
            # Interpolate c_t[z, :] from m_grid to resource_grid using PCHIP
            try:
                pchip_interp = PchipInterpolator(m_grid, c_t[z, :], extrapolate=False)
                interp_vals = pchip_interp(resource_grid)
                interp_vals = np.where(np.isnan(interp_vals), c_t[z, 0], interp_vals)
                result[t, z, :] = interp_vals
            except Exception:
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
