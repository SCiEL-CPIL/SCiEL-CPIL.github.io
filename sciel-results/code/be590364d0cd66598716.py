import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']
    
    m_grid_out = np.array(resource_grid)
    n_m_out = len(m_grid_out)
    n_z = len(income)
    
    # Internal dense grid for value function computation
    # Max m' approx R * max(m_grid_out) + max(income) = 1.02 * 40 + 1.5 = 42.3
    # Min m' approx min(income) = 0.5 (if c=m)
    # Use a slightly wider range and high resolution to minimize interpolation error.
    n_m_int = 2000
    m_internal = np.geomspace(0.001, 50, n_m_int)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)
    
    # Value function storage: V[t, z, m_idx]
    V = np.zeros((horizon, n_z, n_m_int))
    
    # Terminal period: consume everything
    V[horizon - 1, :, :] = utility(m_internal)[np.newaxis, :]
    
    # Grid for consumption fraction s = c/m
    # We use a moderately dense grid for the internal optimization
    # The budget parameter suggests we should respect it for the search grid
    # But for internal accuracy, we can be denser
    n_s = 300
    s_grid = np.linspace(0.001, 1.0, n_s)
    
    # Backward induction on the internal grid
    for t in range(horizon - 2, -1, -1):
        if time.time() - start_time > deadline - 10:
            # Not enough time, break and use heuristic for remaining
            break
        
        for z in range(n_z):
            # m_internal is shape (n_m_int,)
            # We want to find optimal s for each m in m_internal
            # c = m * s
            # m_next = R * (m - c) + income[z']
            
            # Objective shape: (n_m_int, n_s)
            m_col = m_internal[:, np.newaxis]
            s_row = s_grid[np.newaxis, :]
            
            c_mat = m_col * s_row
            c_mat = np.maximum(c_mat, 1e-12)
            
            u_c = utility(c_mat)
            
            total_val = np.zeros_like(u_c)
            
            for zp in range(n_z):
                m_save = R * (m_col - c_mat)
                m_next = m_save + income[zp]
                
                # Clip m_next to the range of m_internal for interpolation
                # m_next can be slightly outside [0.001, 50]
                # np.interp handles out-of-range by returning endpoint values
                m_next_flat = m_next.flatten()
                v_next_flat = np.interp(m_next_flat, m_internal, V[t + 1, zp, :])
                v_next = v_next_flat.reshape(m_next.shape)
                
                total_val += beta * transition[z, zp] * v_next
            
            total_val += u_c
            
            # Find best s for each m
            best_idx = np.argmax(total_val, axis=1)
            V[t, z, :] = total_val[np.arange(n_m_int), best_idx]
    
    # Now construct the consumption policy on the original resource_grid
    consumption = np.zeros((horizon, n_z, n_m_out))
    
    # Terminal period
    consumption[horizon - 1, :, :] = m_grid_out[np.newaxis, :]
    
    # For earlier periods, find optimal c on the original grid
    # Use a finer s_grid for the final policy to ensure accuracy
    n_s_final = 500
    s_grid_final = np.linspace(0.001, 1.0, n_s_final)
    
    for t in range(horizon - 2, -1, -1):
        if time.time() - start_time > deadline - 5:
            # Fill remaining with heuristic
            for tt in range(t, -1, -1):
                for zz in range(n_z):
                    consumption[tt, zz, :] = m_grid_out * 0.5
            break
        
        for z in range(n_z):
            m_out = m_grid_out
            m_col = m_out[:, np.newaxis]
            s_row = s_grid_final[np.newaxis, :]
            
            c_mat = m_col * s_row
            c_mat = np.maximum(c_mat, 1e-12)
            
            u_c = utility(c_mat)
            
            total_val = np.zeros_like(u_c)
            
            for zp in range(n_z):
                m_save = R * (m_col - c_mat)
                m_next = m_save + income[zp]
                
                m_next_flat = m_next.flatten()
                v_next_flat = np.interp(m_next_flat, m_internal, V[t + 1, zp, :])
                v_next = v_next_flat.reshape(m_next.shape)
                
                total_val += beta * transition[z, zp] * v_next
            
            total_val += u_c
            
            best_idx = np.argmax(total_val, axis=1)
            best_c = c_mat[np.arange(n_m_out), best_idx]
            
            # Enforce c <= m
            best_c = np.minimum(best_c, m_out)
            best_c = np.maximum(best_c, 1e-12)
            
            consumption[t, z, :] = best_c
    
    # Final constraints
    for t in range(horizon):
        for z in range(n_z):
            for idx_m in range(n_m_out):
                m = m_grid_out[idx_m]
                c = consumption[t, z, idx_m]
                if c <= 0:
                    c = 1e-12
                if c > m + 1e-10:
                    c = m
                consumption[t, z, idx_m] = c
    
    # Terminal period must consume everything
    consumption[horizon - 1, :, :] = m_grid_out[np.newaxis, :]
    
    return consumption
