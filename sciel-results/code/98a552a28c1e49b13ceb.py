import numpy as np
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_m = len(resource_grid)
    n_z = 2
    
    # Precompute utility and its properties
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
            
    # Value function V[t, z, m_index]
    # V[t] corresponds to the value at time t, given resources m and state z.
    # We compute backwards from t = horizon-1 down to 0.
    V = np.zeros((horizon, n_z, n_m))
    c_opt = np.zeros((horizon, n_z, n_m))
    
    # Last period: consume everything
    for z in range(n_z):
        V[horizon - 1, z, :] = utility(resource_grid)
        c_opt[horizon - 1, z, :] = resource_grid
        
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            for i_m in range(n_m):
                m = resource_grid[i_m]
                
                def objective(c):
                    # c is consumption
                    if c < 0 or c > m:
                        return -np.inf
                    c = max(c, 1e-12)
                    u_c = utility(c)
                    
                    # Expected future value
                    exp_future_val = 0.0
                    for zp in range(n_z):
                        m_next = R * (m - c) + income[zp]
                        # Clip m_next to grid range for interpolation
                        # If m_next is out of bounds, clip to nearest grid point
                        # Note: m_next >= R*0 + min(income) > 0 usually, but let's be safe.
                        # min income is 0.5, R=1.02, so min m_next ~ 0.5. Grid starts at 0.001.
                        # Max m_next: R*40 + 1.5 ~ 42. Grid ends at 40.
                        m_next_clipped = np.clip(m_next, resource_grid[0], resource_grid[-1])
                        
                        # Interpolate V[t+1, zp, :]
                        v_next = np.interp(m_next_clipped, resource_grid, V[t+1, zp, :])
                        exp_future_val += transition[z, zp] * v_next
                        
                    return u_c + beta * exp_future_val
                
                # Maximize objective in [0, m]
                # Use bounded scalar minimization of -objective
                res = minimize_scalar(lambda c: -objective(c), bounds=(1e-12, m), method='bounded', options={'xatol': 1e-8})
                c_star = res.x
                V[t, z, i_m] = -res.fun
                c_opt[t, z, i_m] = c_star
                
    return c_opt
