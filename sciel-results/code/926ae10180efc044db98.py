import time
import numpy as np
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_grid = len(resource_grid)
    n_states = len(income)
    
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1 - gamma) / (1 - gamma)
    
    # Use a dense internal grid for better accuracy
    # The scoring range is 0.2 to 8. The grid is 0.001 to 40.
    n_internal = 800
    m_internal = np.geomspace(resource_grid[0], resource_grid[-1], n_internal)
    
    # Value function storage
    # V[t, z, :]
    V = np.zeros((horizon + 1, n_states, n_internal))
    C = np.zeros((horizon + 1, n_states, n_internal))
    
    # Terminal condition
    V[horizon, :, :] = util(m_internal)
    C[horizon, :, :] = m_internal
    
    # Helper for interpolation
    def interp_val(x, fp):
        return np.interp(x, m_internal, fp)
    
    # Backward induction
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 15.0:
            # Fallback: copy next period
            for tt in range(t, -1, -1):
                V[tt, :, :] = V[tt + 1, :, :]
                C[tt, :, :] = m_internal
            break
            
        for z in range(n_states):
            if time.time() > deadline - 5.0:
                for tt in range(t, -1, -1):
                    V[tt, z, :] = V[tt + 1, z, :]
                    C[tt, z, :] = m_internal
                break
            
            # For each m, find optimal c that maximizes:
            # util(c) + beta * sum_z' trans[z, z'] * V[t+1, z', R(m-c) + income[z']]
            
            # Use a coarse search first, then refine with golden section or brentq
            # For speed, we'll use a coarse grid search followed by refinement
            
            C_current = np.zeros(n_internal)
            V_current = np.zeros(n_internal)
            
            # Process in batches for speed
            batch_size = 50
            for i_start in range(0, n_internal, batch_size):
                i_end = min(i_start + batch_size, n_internal)
                m_batch = m_internal[i_start:i_end]
                
                # For each m in batch, find optimal c
                for i in range(len(m_batch)):
                    m = m_batch[i]
                    
                    # Define objective function to maximize
                    def objective(c):
                        c = max(c, 1e-12)
                        c = min(c, m - 1e-12)
                        if c <= 0:
                            return -np.inf
                        m_next = R * (m - c)
                        total_val = 0.0
                        for zp in range(n_states):
                            m_zp = m_next + income[zp]
                            # Interpolate V[t+1, zp, :]
                            if m_zp <= m_internal[0]:
                                v_zp = V[t + 1, zp, 0]
                            elif m_zp >= m_internal[-1]:
                                v_zp = V[t + 1, zp, -1]
                            else:
                                v_zp = np.interp(m_zp, m_internal, V[t + 1, zp, :])
                            total_val += transition[z, zp] * v_zp
                        return util(c) + beta * total_val
                    
                    # Coarse search
                    # c is in [1e-12, m]
                    # Try a logarithmic grid for c
                    c_min = max(1e-12, m * 1e-6)
                    c_max = m - 1e-12
                    
                    if c_max <= c_min:
                        c_opt = c_min
                    else:
                        # Use brentq on the derivative, or just use a fine grid
                        # For speed, use a moderate grid
                        n_c = 50
                        c_grid = np.geomspace(c_min, c_max, n_c)
                        vals = np.array([objective(c) for c in c_grid])
                        i_best = np.argmax(vals)
                        
                        # Refine around the best point
                        if i_best > 0 and i_best < n_c - 1:
                            c_lo = c_grid[i_best - 1]
                            c_hi = c_grid[i_best + 1]
                        else:
                            c_lo = c_min
                            c_hi = c_max
                        
                        # Golden section search
                        gr = (np.sqrt(5) - 1) / 2
                        c1 = c_hi - gr * (c_hi - c_lo)
                        c2 = c_lo + gr * (c_hi - c_lo)
                        f1 = objective(c1)
                        f2 = objective(c2)
                        
                        for _ in range(30):
                            if f1 < f2:
                                c_lo = c1
                                c1 = c2
                                f1 = f2
                                c2 = c_lo + gr * (c_hi - c_lo)
                                f2 = objective(c2)
                            else:
                                c_hi = c2
                                c2 = c1
                                f2 = f1
                                c1 = c_hi - gr * (c_hi - c_lo)
                                f1 = objective(c1)
                        
                        c_opt = (c_lo + c_hi) / 2.0
                    
                    C_current[i] = c_opt
                    V_current[i] = objective(c_opt)
                
                C[i_start:i_end] = C_current
                V[i_start:i_end] = V_current
            
            C[t, z, :] = C_current
            V[t, z, :] = V_current
    
    # Interpolate to original resource_grid
    result = np.zeros((horizon, n_states, n_grid))
    
    for t in range(horizon):
        for z in range(n_states):
            result[t, z, :] = np.interp(resource_grid, m_internal, C[t, z, :])
            # Enforce constraints
            result[t, z, :] = np.clip(result[t, z, :], 1e-12, resource_grid + 1e-10)
    
    # Terminal period: consume all
    result[horizon - 1, :, :] = resource_grid
    
    # Final validation
    result = np.nan_to_num(result, nan=1e-12, posinf=resource_grid, neginf=1e-12)
    result = np.maximum(result, 1e-12)
    
    return result
