"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    
    # Precompute utility function
    def u(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c**(1 - gamma) / (1 - gamma)
    
    # Store value functions for each time and state
    # V[t, z, m_index]
    V = np.zeros((horizon + 1, n_states, n_grid))
    
    # Policy: c[t, z, m_index]
    C = np.zeros((horizon, n_states, n_grid))
    
    # Terminal condition: consume everything
    for z in range(n_states):
        V[horizon, z] = u(resource_grid)
    
    # Backward induction
    for t in range(horizon - 1, -1, -1):
        for z in range(n_states):
            # For each resource level, find optimal consumption
            for i, m in enumerate(resource_grid):
                # Objective: u(c) + beta * sum_z' P(z'|z) V[t+1, z', R*(m-c) + income[z']]
                # We search over c in (0, m]
                
                def neg_obj(c):
                    if c <= 0 or c > m + 1e-10:
                        return 0.0
                    c = max(c, 1e-12)
                    m_next = R * (m - c)
                    if m_next < 0:
                        m_next = 0.0
                    
                    cont_val = 0.0
                    for zp in range(n_states):
                        p = transition[z, zp]
                        if p > 0:
                            m_next_zp = m_next + income[zp]
                            if m_next_zp <= resource_grid[0]:
                                v_cont = V[t+1, zp, 0]
                            elif m_next_zp >= resource_grid[-1]:
                                v_cont = V[t+1, zp, -1]
                            else:
                                v_cont = np.interp(m_next_zp, resource_grid, V[t+1, zp])
                            cont_val += p * v_cont
                    
                    return -(u(c) + beta * cont_val)
                
                # Coarse search first with more points for better initial bracket
                c_coarse = np.linspace(1e-6, m, 100)
                vals = np.array([neg_obj(c) for c in c_coarse])
                best_idx = np.argmin(vals)
                
                # Local refinement
                if best_idx > 0 and best_idx < len(c_coarse) - 1:
                    lower = c_coarse[best_idx - 1]
                    upper = c_coarse[best_idx + 1]
                else:
                    lower = c_coarse[0]
                    upper = c_coarse[-1]
                
                res = minimize_scalar(neg_obj, bounds=(lower, upper), method='bounded', 
                                       options={'xatol': 1e-12, 'maxiter': 200})
                
                if res.success:
                    c_opt = res.x
                else:
                    c_opt = c_coarse[best_idx]
                
                # Ensure constraints
                c_opt = np.clip(c_opt, 1e-12, m)
                C[t, z, i] = c_opt
                
                # Compute value
                m_next = R * (m - c_opt)
                if m_next < 0:
                    m_next = 0.0
                cont_val = 0.0
                for zp in range(n_states):
                    p = transition[z, zp]
                    if p > 0:
                        m_next_zp = m_next + income[zp]
                        if m_next_zp <= resource_grid[0]:
                            v_cont = V[t+1, zp, 0]
                        elif m_next_zp >= resource_grid[-1]:
                            v_cont = V[t+1, zp, -1]
                        else:
                            v_cont = np.interp(m_next_zp, resource_grid, V[t+1, zp])
                        cont_val += p * v_cont
                
                V[t, z, i] = u(c_opt) + beta * cont_val
        
        # Check time
        if time.time() > deadline:
            # Fill remaining with heuristic: consume proportionally
            for tt in range(t, horizon):
                for zz in range(n_states):
                    for ii in range(n_grid):
                        m_val = resource_grid[ii]
                        frac = 1.0 / (1.0 + beta * R)
                        c_val = frac * m_val
                        c_val = np.clip(c_val, 1e-12, m_val)
                        C[tt, zz, ii] = c_val
            # Ensure terminal condition
            C[horizon-1] = resource_grid[None, :]
            return C

    # Ensure terminal period: consume all
    C[horizon - 1] = resource_grid[None, :]
    
    # Validate: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_states):
            for i in range(n_grid):
                if C[t, z, i] <= 0:
                    C[t, z, i] = 1e-12
                if C[t, z, i] > resource_grid[i] + 1e-10:
                    C[t, z, i] = resource_grid[i]
    
    return C
# EVOLVE-BLOCK-END
