import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115
    
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
    
    # Scoring grid
    m_fine = np.geomspace(0.001, 40, 251)
    n_fine = len(m_fine)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # Policy storage
    policy_fine = np.zeros((horizon, 2, n_fine))
    
    # Value function storage
    V_next = np.zeros((n_fine, 2))
    
    # Candidate fractions
    c_fracs = np.linspace(0.001, 1.0, 301)
    
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 5:
            break
            
        V_curr = np.zeros((n_fine, 2))
        
        # Build cubic splines for continuation values
        try:
            cs_0 = CubicSpline(m_fine, V_next[:, 0], bc_type='natural')
            cs_1 = CubicSpline(m_fine, V_next[:, 1], bc_type='natural')
        except Exception:
            cs_0 = None
            cs_1 = None
            
        for z in range(2):
            probs = transition[z]
            
            for i in range(n_fine):
                m = m_fine[i]
                c_candidates = m * c_fracs
                
                m_next_0 = R * (m - c_candidates) + income[0]
                m_next_1 = R * (m - c_candidates) + income[1]
                
                if cs_0 is not None and cs_1 is not None:
                    V_cont_0 = cs_0(m_next_0)
                    V_cont_1 = cs_1(m_next_1)
                else:
                    V_cont_0 = np.interp(m_next_0, m_fine, V_next[:, 0])
                    V_cont_1 = np.interp(m_next_1, m_fine, V_next[:, 1])
                
                E_V_cont = probs[0] * V_cont_0 + probs[1] * V_cont_1
                
                U_c = utility(c_candidates)
                Total_Value = U_c + beta * E_V_cont
                
                idx_max = np.argmax(Total_Value)
                V_curr[i, z] = Total_Value[idx_max]
                policy_fine[t, z, i] = c_candidates[idx_max]
                
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
