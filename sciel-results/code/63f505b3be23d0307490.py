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
    n_z = len(income)
    
    # Initialize policy and value arrays
    # policy[t, z, i]
    policy = np.zeros((horizon, n_z, n_m))
    # V[t, z, i]
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
        
    # Backward induction
    # We need V_{t+1} to compute V_t
    # We will store V_next as a 2D array (z, m)
    
    for t in range(horizon - 2, -1, -1):
        # V_next is V[t+1]
        V_next = V[t + 1] # shape (n_z, n_m)
        
        for z in range(n_z):
            # Get transition probabilities from z
            probs = transition[z]
            
            for i_m, m in enumerate(resource_grid):
                # Define the objective function to maximize
                # J(c) = u(c) + beta * sum_z' P(z'|z) * V_{t+1}(R(m-c) + y_{z'}, z')
                
                def objective(c):
                    # c is scalar
                    m_prime_0 = R * (m - c) + income[0]
                    m_prime_1 = R * (m - c) + income[1]
                    
                    # Interpolate V_next
                    # np.interp(x, xp, fp)
                    v_0 = np.interp(m_prime_0, resource_grid, V_next[0])
                    v_1 = np.interp(m_prime_1, resource_grid, V_next[1])
                    
                    cont_value = probs[0] * v_0 + probs[1] * v_1
                    return utility(c) + beta * cont_value
                
                # Maximize objective(c) in [1e-12, m]
                # Since the problem is concave, we can use bounded scalar minimization of -objective
                # Or use the FOC directly with brentq if we can compute the derivative easily.
                # Using minimize_scalar is safer against numerical issues in derivative.
                
                # Bounds: c must be > 0 and <= m
                lower_bound = 1e-12
                upper_bound = m
                
                # If m is very small, the interval is tiny.
                if upper_bound - lower_bound < 1e-15:
                    c_opt = upper_bound
                else:
                    res = minimize_scalar(lambda c: -objective(c), 
                                          bounds=(lower_bound, upper_bound), 
                                          method='bounded', 
                                          options={'xatol': 1e-10})
                    c_opt = res.x
                    # Ensure c_opt is within bounds and valid
                    if not np.isfinite(c_opt) or c_opt < lower_bound or c_opt > upper_bound:
                        c_opt = upper_bound # Fallback to consuming all if optimization fails
                    
                policy[t, z, i_m] = c_opt
                V[t, z, i_m] = objective(c_opt)
                
    return policy
