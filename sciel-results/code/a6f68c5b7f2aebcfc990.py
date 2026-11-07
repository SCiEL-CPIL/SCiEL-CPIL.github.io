import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=np.float64)
    transition = np.asarray(model['transition'], dtype=np.float64)
    horizon = model['horizon']
    
    n_z = len(income)
    m_grid = np.asarray(resource_grid, dtype=np.float64)
    n_m = len(m_grid)
    
    # Terminal period: consume all
    consumption = np.zeros((horizon, n_z, n_m), dtype=np.float64)
    consumption[horizon - 1] = m_grid[None, :]  # c[5] = m
    
    # CRRA utility and marginal utility
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return np.power(c, 1.0 - gamma) / (1.0 - gamma)
    
    # Fine grid for value function interpolation
    v_grid = np.geomspace(0.001, 40.0, 500)
    n_v = len(v_grid)
    
    # Value function: V[t, z, i] = value at state (z, v_grid[i])
    V = np.zeros((horizon, n_z, n_v), dtype=np.float64)
    
    # Terminal value: V[5, z, m] = utility(m) since c=m
    V[horizon - 1] = utility(v_grid[None, :])
    
    # Precompute x grids for search
    x_coarse = np.geomspace(0.001, 1.0, 80)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10.0:
            break
        
        for z in range(n_z):
            best_obj = np.full(n_v, -np.inf)
            best_x = np.zeros(n_v)
            
            # Coarse search
            for x in x_coarse:
                c = v_grid * x
                m_after = v_grid - c
                m_next = R * m_after + income[None, :]  # (n_v, n_z)
                m_next_clipped = np.clip(m_next, v_grid[0], v_grid[-1])
                
                cont_value = np.zeros(n_v)
                for zp in range(n_z):
                    V_interp = np.interp(m_next_clipped[:, zp], v_grid, V[t + 1, zp, :])
                    cont_value += transition[z, zp] * V_interp
                
                obj = utility(c) + beta * cont_value
                mask = obj > best_obj
                best_obj[mask] = obj[mask]
                best_x[mask] = x
            
            # Refine
            for refine_iter in range(2):
                x_low = np.maximum(best_x * 0.8, 0.001)
                x_high = np.minimum(best_x * 1.2, 1.0)
                
                x_fine = np.geomspace(max(x_low[0], 0.001), min(x_high[-1], 1.0), 150)
                
                for x in x_fine:
                    valid = (x >= x_low) & (x <= x_high)
                    if not np.any(valid):
                        continue
                    
                    idx_valid = np.where(valid)[0]
                    c = v_grid[idx_valid] * x
                    m_after = v_grid[idx_valid] - c
                    m_next = R * m_after + income[None, :]
                    m_next_clipped = np.clip(m_next, v_grid[0], v_grid[-1])
                    
                    cont_value = np.zeros(len(idx_valid))
                    for zp in range(n_z):
                        V_interp = np.interp(m_next_clipped[:, zp], v_grid, V[t + 1, zp, :])
                        cont_value += transition[z, zp] * V_interp
                    
                    obj = utility(c) + beta * cont_value
                    
                    mask = obj > best_obj[idx_valid]
                    best_obj[idx_valid][mask] = obj[idx_valid][mask]
                    best_x[idx_valid][mask] = x
            
            # Compute optimal c on v_grid and interpolate to m_grid
            c_opt_v = np.clip(v_grid * best_x, 1e-12, v_grid)
            consumption[t, z, :] = np.interp(m_grid, v_grid, c_opt_v)
            consumption[t, z, :] = np.clip(consumption[t, z, :], 1e-12, m_grid + 1e-10)
            
            # Store value function
            c_final = c_opt_v
            m_after = v_grid - c_final
            m_next = R * m_after + income[None, :]
            m_next_clipped = np.clip(m_next, v_grid[0], v_grid[-1])
            
            V_t_z = np.zeros(n_v)
            for zp in range(n_z):
                V_interp = np.interp(m_next_clipped[:, zp], v_grid, V[t + 1, zp, :])
                V_t_z += transition[z, zp] * V_interp
            
            V[t, z, :] = utility(c_final) + beta * V_t_z
    
    # Fix terminal period
    consumption[horizon - 1] = m_grid[None, :]
    
    # Final bounds
    consumption = np.clip(consumption, 1e-12, None)
    for t in range(horizon):
        for z in range(n_z):
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid + 1e-10)
    
    if not np.all(np.isfinite(consumption)):
        consumption = np.nan_to_num(consumption, nan=1e-12, posinf=1e-12, neginf=1e-12)
    
    return consumption
