"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def solve_consumption(model, resource_grid, budget):
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_z = len(income)
    n_m = len(resource_grid)
    
    c = np.zeros((horizon, n_z, n_m))
    c[-1] = resource_grid[np.newaxis, :]
    
    def fit_loglinear(m_vals, v_vals):
        log_m = np.log(np.maximum(m_vals, 1e-12))
        A_mat = np.column_stack([np.ones_like(log_m), log_m])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, v_vals, rcond=None)
        return coeffs[0], coeffs[1]
    
    A = np.zeros(n_z)
    B = np.zeros(n_z)
    
    for z in range(n_z):
        m_vals = resource_grid
        if gamma == 1:
            v_vals = np.log(np.maximum(m_vals, 1e-12))
        else:
            v_vals = np.power(np.maximum(m_vals, 1e-12), 1-gamma) / (1-gamma)
        A[z], B[z] = fit_loglinear(m_vals, v_vals)
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break
        
        A_new = np.zeros(n_z)
        B_new = np.zeros(n_z)
        
        for z in range(n_z):
            m_vals = resource_grid
            c_iter = m_vals * 0.5
            c_iter = np.maximum(c_iter, 1e-12)
            
            for _nk in range(50):
                if time.time() > deadline:
                    break
                
                m_prime = R * (m_vals[:, np.newaxis] - c_iter[:, np.newaxis]) + income[np.newaxis, :]
                m_prime = np.maximum(m_prime, 1e-12)
                
                marginal_value = np.zeros_like(m_vals)
                for zp in range(n_z):
                    marginal_value += transition[z, zp] * B[zp] / m_prime[:, zp]
                marginal_value *= beta * R
                
                if gamma == 1:
                    u_prime = 1.0 / c_iter
                else:
                    u_prime = np.power(c_iter, -gamma)
                
                residual = u_prime - marginal_value
                
                if gamma == 1:
                    du_prime_dc = -1.0 / (c_iter ** 2)
                else:
                    du_prime_dc = -gamma * np.power(c_iter, -gamma - 1)
                
                dmarg_dc = np.zeros_like(m_vals)
                for zp in range(n_z):
                    dmarg_dc += transition[z, zp] * B[zp] * (-R) / (m_prime[:, zp] ** 2)
                dmarg_dc *= beta * R
                
                dresidual_dc = du_prime_dc - dmarg_dc
                
                step = residual / dresidual_dc
                c_new = c_iter - step
                c_new = np.clip(c_new, 1e-12, m_vals)
                
                if np.max(np.abs(c_new - c_iter)) < 1e-10:
                    c_iter = c_new
                    break
                c_iter = c_new
            
            c[t, z, :] = c_iter
            
            c_star = c_iter
            m_prime = R * (m_vals[:, np.newaxis] - c_star[:, np.newaxis]) + income[np.newaxis, :]
            m_prime = np.maximum(m_prime, 1e-12)
            
            if gamma == 1:
                u_c = np.log(np.maximum(c_star, 1e-12))
            else:
                u_c = np.power(np.maximum(c_star, 1e-12), 1-gamma) / (1-gamma)
            
            V_vals = u_c.copy()
            for zp in range(n_z):
                V_vals += beta * transition[z, zp] * (A[zp] + B[zp] * np.log(m_prime[:, zp]))
            
            A_new[z], B_new[z] = fit_loglinear(m_vals, V_vals)
        
        A, B = A_new, B_new
    
    c[-1] = resource_grid[np.newaxis, :]
    c = np.clip(c, 1e-12, None)
    for t in range(horizon):
        for z in range(n_z):
            c[t, z, :] = np.minimum(c[t, z, :], resource_grid + 1e-10)
    
    return c
# EVOLVE-BLOCK-END
