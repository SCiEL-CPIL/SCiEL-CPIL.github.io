"""Redesigned: piecewise log-linear value function with analytical CRRA Euler solve."""
import numpy as np

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    n_intervals = n_grid - 1
    
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    def mu_inv(mu_val):
        mu_val = np.maximum(mu_val, 1e-15)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / mu_val
        else:
            return np.power(mu_val, -1.0 / gamma)
    
    log_grid = np.log(resource_grid)
    
    # Terminal V: V_{H-1}(m, z) = u(m)
    V = np.zeros((n_states, n_grid))
    for z in range(n_states):
        V[z, :] = util(resource_grid)
    
    consumption = np.zeros((horizon, n_states, n_grid))
    consumption[horizon-1, :, :] = resource_grid
    
    dlog = log_grid[1:] - log_grid[:-1]
    
    for t in range(horizon - 2, -1, -1):
        V_curr = np.zeros((n_states, n_grid))
        
        # Compute slopes and intercepts for current V (piecewise linear in log m)
        slope = np.zeros((n_states, n_intervals))
        intercept = np.zeros((n_states, n_intervals))
        for zp in range(n_states):
            slope[zp, :] = (V[zp, 1:] - V[zp, :-1]) / dlog
            intercept[zp, :] = V[zp, :-1] - slope[zp, :] * log_grid[:-1]
        
        for z in range(n_states):
            m_vals = resource_grid
            c = 0.5 * m_vals
            
            for _ in range(20):
                exp_dV_dm = np.zeros(n_grid)
                exp_V = np.zeros(n_grid)
                
                for zp in range(n_states):
                    p = transition[z, zp]
                    if p < 1e-15:
                        continue
                    
                    m_prime = R * (m_vals - c) + income[zp]
                    m_prime = np.clip(m_prime, resource_grid[0], resource_grid[-1])
                    
                    log_mp = np.log(m_prime)
                    i_prime = np.searchsorted(log_grid, log_mp) - 1
                    i_prime = np.clip(i_prime, 0, n_intervals - 1)
                    
                    s_zp = slope[zp, i_prime]
                    a_zp = intercept[zp, i_prime]
                    
                    dV_dm_prime = s_zp / m_prime
                    V_prime = a_zp + s_zp * log_mp
                    
                    exp_dV_dm += p * dV_dm_prime
                    exp_V += p * V_prime
                
                rhs = beta * R * exp_dV_dm
                c_new = mu_inv(rhs)
                c_new = np.clip(c_new, 1e-12, m_vals)
                
                if np.max(np.abs(c_new - c)) < 1e-12 * np.max(m_vals):
                    c = c_new
                    break
                c = c_new
            
            consumption[t, z, :] = c
            V_curr[z, :] = util(c) + beta * exp_V
        
        V = V_curr
    
    consumption[horizon-1, :, :] = resource_grid
    
    for t in range(horizon):
        for z in range(n_states):
            consumption[t, z, :] = np.clip(consumption[t, z, :], 1e-12, resource_grid)
    
    return consumption
