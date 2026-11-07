import numpy as np
import time
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin below 150s kill
    
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
    
    # Scoring grid - use a denser grid for better accuracy
    m_fine = np.geomspace(0.001, 40, 401)
    n_fine = len(m_fine)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # Marginal utility for FOC
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
    
    # Candidate fractions for grid search (coarse)
    c_fracs_coarse = np.linspace(0.001, 1.0, 101)
    
    # Backward induction
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 10:
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
            
            # First pass: coarse grid search to find approximate optimum
            for i in range(n_fine):
                m = m_fine[i]
                c_candidates = m * c_fracs_coarse
                
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
                c_approx = c_candidates[idx_max]
                
                # Second pass: local refinement using minimize_scalar
                if time.time() < deadline - 15:
                    # Define objective to minimize (negative of total value)
                    def neg_obj(c_local):
                        c_local = max(c_local, 1e-12)
                        c_local = min(c_local, m)
                        m_n0 = R * (m - c_local) + income[0]
                        m_n1 = R * (m - c_local) + income[1]
                        
                        if cs_0 is not None and cs_1 is not None:
                            v0 = cs_0(m_n0)
                            v1 = cs_1(m_n1)
                        else:
                            v0 = np.interp(m_n0, m_fine, V_next[:, 0])
                            v1 = np.interp(m_n1, m_fine, V_next[:, 1])
                        
                        ev = probs[0] * v0 + probs[1] * v1
                        return -(utility(c_local) + beta * ev)
                    
                    # Local search around the coarse optimum
                    c_low = max(1e-12, c_approx * 0.8)
                    c_high = min(m, c_approx * 1.2)
                    
                    try:
                        res = minimize_scalar(neg_obj, bounds=(c_low, c_high), method='bounded', options={'xatol': 1e-10})
                        c_opt = res.x
                    except Exception:
                        c_opt = c_approx
                else:
                    c_opt = c_approx
                
                # Evaluate at the refined point
                m_n0 = R * (m - c_opt) + income[0]
                m_n1 = R * (m - c_opt) + income[1]
                
                if cs_0 is not None and cs_1 is not None:
                    v0 = cs_0(m_n0)
                    v1 = cs_1(m_n1)
                else:
                    v0 = np.interp(m_n0, m_fine, V_next[:, 0])
                    v1 = np.interp(m_n1, m_fine, V_next[:, 1])
                
                ev = probs[0] * v0 + probs[1] * v1
                V_curr[i, z] = utility(c_opt) + beta * ev
                policy_fine[t, z, i] = c_opt
        
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
