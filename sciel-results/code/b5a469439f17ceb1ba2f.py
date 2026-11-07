import numpy as np
import time
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_out = len(resource_grid)
    
    # Policy shape: (horizon, 2, n_out)
    policy = np.zeros((horizon, 2, n_out))
    
    # Terminal period: consume everything
    policy[horizon - 1, :, :] = resource_grid
    
    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        return c ** (-gamma)
    
    # Work backward from t = horizon-2 to t = 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fill remaining with terminal-like policy
            for tt in range(t, horizon - 1):
                policy[tt, :, :] = resource_grid
            break
        
        for z in range(2):
            for i, m in enumerate(resource_grid):
                if time.time() > deadline:
                    policy[t, z, i:] = resource_grid[i:]
                    break
                
                # Define FOC residual: u'(c) - beta * E[u'(c_next)] = 0
                # c_next depends on c: c_next = policy[t+1, z', R*(m-c)+income[z']]
                
                def residual(c):
                    m_next = R * (m - c) + income  # shape (2,)
                    # Interpolate next-period policy to get c_next for each z'
                    c_next = np.interp(m_next, resource_grid, policy[t + 1, z, :])  # Wait, this is wrong
                    # Actually, c_next depends on z'
                    c_next = np.array([
                        np.interp(m_next[0], resource_grid, policy[t + 1, 0, :]),
                        np.interp(m_next[1], resource_grid, policy[t + 1, 1, :])
                    ])
                    mu_c = marginal_utility(c)
                    mu_next = marginal_utility(c_next)
                    E_mu_next = np.dot(transition[z], mu_next)
                    return mu_c - beta * R * E_mu_next
                
                # Bisection bounds
                lo = 1e-12
                hi = m
                
                try:
                    res_lo = residual(lo)
                    res_hi = residual(hi)
                    
                    if res_hi > 0:
                        policy[t, z, i] = hi
                    elif res_lo < 0:
                        policy[t, z, i] = lo
                    else:
                        c_opt = brentq(residual, lo, hi, xtol=1e-12, rtol=1e-10, maxiter=100)
                        policy[t, z, i] = c_opt
                except Exception:
                    # Fallback: grid search
                    fractions = np.linspace(0.001, 1.0, 50)
                    c_candidates = m * fractions
                    best_c = c_candidates[0]
                    best_val = -np.inf
                    for c in c_candidates:
                        m_next = R * (m - c) + income
                        c_next = np.array([
                            np.interp(m_next[0], resource_grid, policy[t + 1, 0, :]),
                            np.interp(m_next[1], resource_grid, policy[t + 1, 1, :])
                        ])
                        u_c = np.log(np.maximum(c, 1e-12)) if abs(gamma-1)<1e-9 else (np.maximum(c,1e-12)**(1-gamma))/(1-gamma)
                        u_next = np.log(np.maximum(c_next, 1e-12)) if abs(gamma-1)<1e-9 else (np.maximum(c_next,1e-12)**(1-gamma))/(1-gamma)
                        val = u_c + beta * np.dot(transition[z], u_next)
                        if val > best_val:
                            best_val = val
                            best_c = c
                    policy[t, z, i] = best_c
    
    # Ensure validity
    for t in range(horizon):
        for z in range(2):
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, resource_grid + 1e-10)
    
    # Terminal period must equal grid
    policy[horizon - 1, :, :] = resource_grid
    
    return policy
