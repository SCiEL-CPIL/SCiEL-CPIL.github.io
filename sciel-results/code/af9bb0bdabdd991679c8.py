"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M = len(resource_grid)
    Z = len(income)
    
    # Finer internal grid
    m_fine = np.geomspace(0.001, 40, 401)
    M_fine = len(m_fine)
    
    # Consumption grid points per m
    n_c = 200
    
    consumption = np.zeros((horizon, Z, M))
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    # V_next[z, :] = V_{t+1} on m_fine
    V_next = np.zeros((Z, M_fine))
    
    for t in range(horizon - 1, -1, -1):
        V_curr = np.zeros((Z, M_fine))
        
        for z in range(Z):
            # Vectorized computation over all m_fine points
            # c_grid shape: (M_fine, n_c)
            # c = m * linspace(0.001, 1, n_c)
            frac = np.linspace(0.001, 1.0, n_c)[None, :]
            c_grid = m_fine[:, None] * frac  # (M_fine, n_c)
            c_grid = np.maximum(c_grid, 1e-12)
            
            # m_prime = R*(m - c) + income[z'], shape: (M_fine, n_c, Z)
            m_prime = R * (m_fine[:, None, None] - c_grid[:, :, None]) + income[None, None, :]
            m_prime_clipped = np.clip(m_prime, m_fine[0], m_fine[-1])
            
            # Interpolate V_next: for each (mi, ci, zp), get V_next[zp, m_prime]
            # We need to do this efficiently
            # V_next shape: (Z, M_fine)
            # For each zp, V_next[zp, :] is a function of m
            # We want cont_val[mi, ci, zp] = V_next[zp, m_prime[mi, ci, zp]]
            
            # Do it per zp
            cont_val = np.zeros((M_fine, n_c, Z))
            for zp in range(Z):
                # m_prime[:, :, zp] shape: (M_fine, n_c)
                # V_next[zp, :] shape: (M_fine,)
                # Need to interpolate for each (mi, ci) pair
                # This is a 2D interpolation, but we can flatten
                mp_flat = m_prime_clipped[:, :, zp].ravel()
                v_flat = np.interp(mp_flat, m_fine, V_next[zp, :])
                cont_val[:, :, zp] = v_flat.reshape(M_fine, n_c)
            
            # exp_cont[mi, ci] = sum_zp transition[z, zp] * cont_val[mi, ci, zp]
            exp_cont = np.tensordot(transition[z, :], cont_val, axes=([0], [2]))  # (M_fine, n_c)
            
            # obj[mi, ci] = utility(c_grid[mi, ci]) + beta * exp_cont[mi, ci]
            obj = utility(c_grid) + beta * exp_cont
            
            # Find best c for each mi
            best_idx = np.argmax(obj, axis=1)  # (M_fine,)
            best_c = c_grid[np.arange(M_fine), best_idx]
            best_val = obj[np.arange(M_fine), best_idx]
            
            V_curr[z, :] = best_val
            
            # Extract policy for resource_grid
            # For each resource_grid point, find optimal c
            # We can interpolate best_c from m_fine to resource_grid
            # But it's better to recompute for accuracy
            if time.time() < deadline - 10:
                for ri in range(M):
                    m_val = resource_grid[ri]
                    c_grid_r = m_val * frac[0, :]  # (n_c,)
                    c_grid_r = np.maximum(c_grid_r, 1e-12)
                    
                    m_prime_r = R * (m_val - c_grid_r[:, None]) + income[None, :]  # (n_c, Z)
                    m_prime_r_clipped = np.clip(m_prime_r, m_fine[0], m_fine[-1])
                    
                    cont_val_r = np.zeros((n_c, Z))
                    for zp in range(Z):
                        cont_val_r[:, zp] = np.interp(m_prime_r_clipped[:, zp], m_fine, V_next[zp, :])
                    
                    exp_cont_r = transition[z, :] @ cont_val_r.T
                    obj_r = utility(c_grid_r) + beta * exp_cont_r
                    
                    best_idx_r = np.argmax(obj_r)
                    consumption[t, z, ri] = c_grid_r[best_idx_r]
            else:
                # Fallback: interpolate
                consumption[t, z, :] = np.interp(resource_grid, m_fine, best_c)
        
        V_next = V_curr
        
        if time.time() > deadline - 5:
            for tt in range(t, -1, -1):
                for z in range(Z):
                    consumption[tt, z, :] = resource_grid
            break
    
    # Terminal condition
    consumption[horizon - 1, :, :] = resource_grid[None, None, :]
    
    # Constraints
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.clip(consumption[t, z, :], 1e-12, resource_grid + 1e-10)
    
    return consumption
# EVOLVE-BLOCK-END
