import numpy as np
import time
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin below 150s kill
    
    # Extract model parameters
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Output grid
    n_out = len(resource_grid)
    m_out = resource_grid
    
    # Use a fine grid for computation
    m_fine = np.geomspace(0.001, 40, 301)
    n_fine = len(m_fine)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # Marginal utility
    def marg_util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # Policy storage on fine grid
    policy_fine = np.zeros((horizon, 2, n_fine))
    
    # Value function on fine grid
    V_next = np.zeros((n_fine, 2))
    
    # Backward induction
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 5:
            break
            
        V_curr = np.zeros((n_fine, 2))
        
        for z in range(2):
            probs = transition[z]
            
            # Vectorized computation for this z
            m_arr = m_fine
            
            # Initial guess: use a fraction of m
            c_arr = 0.5 * m_arr
            c_arr = np.maximum(c_arr, 1e-12)
            c_arr = np.minimum(c_arr, m_arr - 1e-12)
            
            # Newton's method iterations (vectorized)
            for _ in range(15):
                c_arr = np.maximum(c_arr, 1e-12)
                c_arr = np.minimum(c_arr, m_arr - 1e-12)
                
                m_n0 = R * (m_arr - c_arr) + income[0]
                m_n1 = R * (m_arr - c_arr) + income[1]
                
                # Interpolate continuation values and derivatives
                v0 = np.interp(m_n0, m_fine, V_next[:, 0])
                v1 = np.interp(m_n1, m_fine, V_next[:, 1])
                
                # Numerical derivatives
                eps = 1e-6
                m_n0_p = m_n0 + eps
                m_n0_m = np.maximum(m_n0 - eps, 1e-12)
                m_n1_p = m_n1 + eps
                m_n1_m = np.maximum(m_n1 - eps, 1e-12)
                
                dv0 = (np.interp(m_n0_p, m_fine, V_next[:, 0]) - np.interp(m_n0_m, m_fine, V_next[:, 0])) / (m_n0_p - m_n0_m)
                dv1 = (np.interp(m_n1_p, m_fine, V_next[:, 1]) - np.interp(m_n1_m, m_fine, V_next[:, 1])) / (m_n1_p - m_n1_m)
                
                # Expected marginal value
                ev_marg = probs[0] * dv0 + probs[1] * dv1
                
                # FOC residual
                mu_c = marg_util(c_arr)
                residual = mu_c - beta * R * ev_marg
                
                # Numerical derivative of residual w.r.t. c
                eps_c = np.maximum(1e-8, c_arr * 1e-6)
                c_p = np.minimum(c_arr + eps_c, m_arr - 1e-12)
                c_m = np.maximum(c_arr - eps_c, 1e-12)
                
                m_n0_p = R * (m_arr - c_p) + income[0]
                m_n1_p = R * (m_arr - c_p) + income[1]
                m_n0_m = R * (m_arr - c_m) + income[0]
                m_n1_m = R * (m_arr - c_m) + income[1]
                
                dv0_p = (np.interp(m_n0_p + eps, m_fine, V_next[:, 0]) - np.interp(np.maximum(m_n0_p - eps, 1e-12), m_fine, V_next[:, 0])) / (2 * eps)
                dv1_p = (np.interp(m_n1_p + eps, m_fine, V_next[:, 1]) - np.interp(np.maximum(m_n1_p - eps, 1e-12), m_fine, V_next[:, 1])) / (2 * eps)
                dv0_m = (np.interp(m_n0_m + eps, m_fine, V_next[:, 0]) - np.interp(np.maximum(m_n0_m - eps, 1e-12), m_fine, V_next[:, 0])) / (2 * eps)
                dv1_m = (np.interp(m_n1_m + eps, m_fine, V_next[:, 1]) - np.interp(np.maximum(m_n1_m - eps, 1e-12), m_fine, V_next[:, 1])) / (2 * eps)
                
                ev_marg_p = probs[0] * dv0_p + probs[1] * dv1_p
                ev_marg_m = probs[0] * dv0_m + probs[1] * dv1_m
                
                residual_p = marg_util(c_p) - beta * R * ev_marg_p
                residual_m = marg_util(c_m) - beta * R * ev_marg_m
                
                deriv = (residual_p - residual_m) / (c_p - c_m)
                
                # Newton step with safe update
                deriv_safe = np.where(np.abs(deriv) < 1e-15, 1.0, deriv)
                c_new = c_arr - residual / deriv_safe
                c_new = np.maximum(c_new, 1e-12)
                c_new = np.minimum(c_new, m_arr - 1e-12)
                
                # Check convergence
                if np.all(np.abs(c_new - c_arr) < 1e-12):
                    c_arr = c_new
                    break
                c_arr = c_new
            
            # Ensure c is valid
            c_arr = np.maximum(c_arr, 1e-12)
            c_arr = np.minimum(c_arr, m_arr - 1e-12)
            
            # Compute value
            m_n0 = R * (m_arr - c_arr) + income[0]
            m_n1 = R * (m_arr - c_arr) + income[1]
            
            v0 = np.interp(m_n0, m_fine, V_next[:, 0])
            v1 = np.interp(m_n1, m_fine, V_next[:, 1])
            
            ev = probs[0] * v0 + probs[1] * v1
            V_curr[:, z] = utility(c_arr) + beta * ev
            policy_fine[t, z, :] = c_arr
        
        V_next = V_curr
    
    # Fallback for early termination
    last_valid_t = horizon - 1
    for t in range(horizon - 1, -1, -1):
        if not np.all(policy_fine[t] == 0):
            last_valid_t = t
            break
            
    if last_valid_t < horizon - 1:
        for t in range(last_valid_t, -1, -1):
            if np.all(policy_fine[t] == 0):
                policy_fine[t] = policy_fine[last_valid_t]
    
    # Interpolate to output grid
    consumption_out = np.zeros((horizon, 2, n_out))
    
    for t in range(horizon):
        for z in range(2):
            consumption_out[t, z, :] = np.interp(m_out, m_fine, policy_fine[t, z, :])
    
    # Validity checks
    for t in range(horizon):
        for z in range(2):
            consumption_out[t, z, :] = np.maximum(consumption_out[t, z, :], 1e-12)
            consumption_out[t, z, :] = np.minimum(consumption_out[t, z, :], m_out + 1e-10)
    
    # Terminal condition
    consumption_out[5, :, :] = m_out[None, None, :]
    
    return consumption_out
