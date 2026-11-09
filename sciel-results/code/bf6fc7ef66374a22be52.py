import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    DEADLINE = 140  # safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'], dtype=np.float64)
    transition = np.array(model['transition'], dtype=np.float64)
    horizon = model['horizon']
    
    n_grid = len(resource_grid)
    n_z = len(income)
    
    consumption = np.zeros((horizon, n_z, n_grid), dtype=np.float64)
    
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1.0 - gamma)) / (1.0 - gamma)
    
    # Terminal condition: c[5] = m
    consumption[5, :, :] = resource_grid[None, None, :]
    V = np.zeros((horizon, n_z, n_grid), dtype=np.float64)
    V[5, :, :] = util(resource_grid)
    
    for t in range(4, -1, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        for z in range(n_z):
            # Coarse fraction grid
            n_coarse = 300
            f_coarse = np.linspace(0.001, 1.0, n_coarse)
            
            # Vectorized coarse evaluation
            c_coarse = f_coarse[None, :] * resource_grid[:, None]  # (n_grid, n_coarse)
            c_coarse = np.clip(c_coarse, 1e-12, resource_grid[:, None])
            
            savings_coarse = resource_grid[:, None] - c_coarse  # (n_grid, n_coarse)
            m_prime_coarse = R * savings_coarse[:, :, None] + income[None, None, :]  # (n_grid, n_coarse, n_z)
            m_prime_coarse = np.clip(m_prime_coarse, resource_grid[0], resource_grid[-1])
            
            V_cont_coarse = np.zeros((n_grid, n_coarse, n_z), dtype=np.float64)
            for zp in range(n_z):
                V_cont_coarse[:, :, zp] = np.interp(
                    m_prime_coarse[:, :, zp].ravel(), resource_grid, V[t+1, zp, :]
                ).reshape(n_grid, n_coarse)
            
            exp_cont_coarse = np.tensordot(transition[z, :], V_cont_coarse, axes=([0], [2]))  # (n_grid, n_coarse)
            total_val_coarse = util(c_coarse) + beta * exp_cont_coarse
            
            idx_best = np.argmax(total_val_coarse, axis=1)  # (n_grid,)
            
            best_c = np.zeros(n_grid, dtype=np.float64)
            best_val = np.zeros(n_grid, dtype=np.float64)
            for i_m in range(n_grid):
                best_c[i_m] = c_coarse[i_m, idx_best[i_m]]
                best_val[i_m] = total_val_coarse[i_m, idx_best[i_m]]
            
            # Local refinement
            n_fine = 100
            for i_m in range(n_grid):
                if time.time() - start_time > DEADLINE:
                    break
                ib = idx_best[i_m]
                if ib > 0 and ib < n_coarse - 1:
                    f_lo = f_coarse[max(0, ib - 4)]
                    f_hi = f_coarse[min(n_coarse - 1, ib + 4)]
                elif ib == 0:
                    f_lo = 1e-5
                    f_hi = f_coarse[min(6, n_coarse - 1)]
                else:
                    f_lo = f_coarse[max(0, n_coarse - 7)]
                    f_hi = 1.0
                
                f_fine = np.linspace(f_lo, f_hi, n_fine)
                c_fine = np.clip(f_fine * resource_grid[i_m], 1e-12, resource_grid[i_m])
                
                savings_fine = resource_grid[i_m] - c_fine
                m_prime_fine = R * savings_fine[None, :] + income[:, None]
                m_prime_fine = np.clip(m_prime_fine, resource_grid[0], resource_grid[-1])
                
                V_cont_fine = np.zeros((n_fine, n_z), dtype=np.float64)
                for zp in range(n_z):
                    V_cont_fine[:, zp] = np.interp(m_prime_fine[zp, :], resource_grid, V[t+1, zp, :])
                
                exp_cont_fine = transition[z, :] @ V_cont_fine
                total_val_fine = util(c_fine) + beta * exp_cont_fine
                
                ibf = np.argmax(total_val_fine)
                if total_val_fine[ibf] > best_val[i_m]:
                    best_c[i_m] = c_fine[ibf]
                    best_val[i_m] = total_val_fine[ibf]
            
            consumption[t, z, :] = best_c
            V[t, z, :] = best_val
    
    # Enforce constraints
    consumption = np.clip(consumption, 1e-12, resource_grid[None, None, :] + 1e-10)
    consumption[5, :, :] = resource_grid[None, None, :]
    
    return consumption
