import numpy as np
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_m = len(resource_grid)
    n_z = len(income)
    
    # Policy and value arrays
    policy = np.zeros((horizon, n_z, n_m))
    V = np.zeros((horizon, n_z, n_m))
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1.0 - gamma)) / (1.0 - gamma)
        
    def utility_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Last period: consume all
    for z in range(n_z):
        policy[horizon - 1, z, :] = resource_grid
        V[horizon - 1, z, :] = utility(resource_grid)
    
    # Precompute interpolation parameters for each z
    # We'll use np.interp for value interpolation
    # For derivative, we'll use central differences on the grid
    
    for t in range(horizon - 2, -1, -1):
        V_next = V[t + 1]  # shape (n_z, n_m)
        
        # Precompute derivative approximations for V_next
        # Using central differences where possible, one-sided at boundaries
        V_next_deriv = np.zeros_like(V_next)
        for z in range(n_z):
            # Central difference for interior points
            V_next_deriv[z, 1:-1] = (V_next[z, 2:] - V_next[z, :-2]) / (resource_grid[2:] - resource_grid[:-2])
            # One-sided at boundaries
            V_next_deriv[z, 0] = (V_next[z, 1] - V_next[z, 0]) / (resource_grid[1] - resource_grid[0])
            V_next_deriv[z, -1] = (V_next[z, -1] - V_next[z, -2]) / (resource_grid[-1] - resource_grid[-2])
        
        for z in range(n_z):
            probs = transition[z]
            
            for i_m, m in enumerate(resource_grid):
                # Define the FOC: u'(c) = beta * R * sum_z' P(z'|z) * V'_{t+1}(R(m-c) + y_{z'}, z')
                # We want to find c such that:
                # u'(c) - beta * R * [probs[0] * V'_next[0](R(m-c)+income[0]) + probs[1] * V'_next[1](R(m-c)+income[1])] = 0
                
                def foc(c):
                    m_prime_0 = R * (m - c) + income[0]
                    m_prime_1 = R * (m - c) + income[1]
                    
                    # Interpolate derivatives
                    # Clip to grid range to avoid extrapolation issues
                    m_prime_0_clipped = np.clip(m_prime_0, resource_grid[0], resource_grid[-1])
                    m_prime_1_clipped = np.clip(m_prime_1, resource_grid[0], resource_grid[-1])
                    
                    v_deriv_0 = np.interp(m_prime_0_clipped, resource_grid, V_next_deriv[0])
                    v_deriv_1 = np.interp(m_prime_1_clipped, resource_grid, V_next_deriv[1])
                    
                    cont_deriv = probs[0] * v_deriv_0 + probs[1] * v_deriv_1
                    
                    return utility_deriv(c) - beta * R * cont_deriv
                
                # Solve for c in (1e-12, m)
                lower = 1e-12
                upper = m
                
                if upper - lower < 1e-15:
                    c_opt = upper
                else:
                    # Check signs at boundaries
                    f_lower = foc(lower)
                    f_upper = foc(upper)
                    
                    if f_lower * f_upper > 0:
                        # No sign change - use boundary
                        if f_lower < 0:
                            c_opt = lower
                        else:
                            c_opt = upper
                    else:
                        try:
                            c_opt = brentq(foc, lower, upper, xtol=1e-12, rtol=1e-10)
                        except:
                            # Fallback: use minimize_scalar
                            from scipy.optimize import minimize_scalar
                            def objective(c):
                                m_prime_0 = R * (m - c) + income[0]
                                m_prime_1 = R * (m - c) + income[1]
                                m_prime_0_clipped = np.clip(m_prime_0, resource_grid[0], resource_grid[-1])
                                m_prime_1_clipped = np.clip(m_prime_1, resource_grid[0], resource_grid[-1])
                                v_0 = np.interp(m_prime_0_clipped, resource_grid, V_next[0])
                                v_1 = np.interp(m_prime_1_clipped, resource_grid, V_next[1])
                                cont_value = probs[0] * v_0 + probs[1] * v_1
                                return utility(c) + beta * cont_value
                            
                            res = minimize_scalar(lambda c: -objective(c), 
                                                  bounds=(lower, upper), 
                                                  method='bounded', 
                                                  options={'xatol': 1e-10})
                            c_opt = res.x
                            if not np.isfinite(c_opt) or c_opt < lower or c_opt > upper:
                                c_opt = upper
                    
                    # Ensure c_opt is valid
                    if not np.isfinite(c_opt) or c_opt < lower or c_opt > upper:
                        c_opt = upper
                
                policy[t, z, i_m] = c_opt
                
                # Compute V[t, z, i_m]
                m_prime_0 = R * (m - c_opt) + income[0]
                m_prime_1 = R * (m - c_opt) + income[1]
                m_prime_0_clipped = np.clip(m_prime_0, resource_grid[0], resource_grid[-1])
                m_prime_1_clipped = np.clip(m_prime_1, resource_grid[0], resource_grid[-1])
                v_0 = np.interp(m_prime_0_clipped, resource_grid, V_next[0])
                v_1 = np.interp(m_prime_1_clipped, resource_grid, V_next[1])
                cont_value = probs[0] * v_0 + probs[1] * v_1
                V[t, z, i_m] = utility(c_opt) + beta * cont_value
    
    return policy
