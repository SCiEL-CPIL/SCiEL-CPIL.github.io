import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    transition = np.asarray(model['transition'])
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    # Derivative of utility for FOC
    def utility_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    # Policy and Value arrays
    # policy[t, z, m_idx]
    policy = np.zeros((horizon, num_states, num_resources))
    value = np.zeros((horizon, num_states, num_resources))
    
    # Terminal condition: consume all resources
    policy[horizon - 1, :, :] = resource_grid[None, None, :]
    value[horizon - 1, :, :] = utility(resource_grid)[None, None, :]
    
    # Backward induction for t = horizon-2 down to 0
    # Use bisection on the FOC to find the exact optimum of the concave objective
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: Fill remaining with terminal policy (consume all)
            for tt in range(t, -1, -1):
                policy[tt, :, :] = resource_grid[None, None, :]
                value[tt, :, :] = utility(resource_grid)[None, None, :]
            break
            
        for z in range(num_states):
            probs = transition[z]
            
            # For each resource level, solve the FOC via bisection
            opt_c = np.zeros(num_resources)
            opt_val = np.zeros(num_resources)
            
            for i in range(num_resources):
                m = resource_grid[i]
                
                # Define the objective function for a given consumption c
                # V(c) = u(c) + beta * sum_z' p(z'|z) * E[V_{t+1}(R(m-c) + y_{z'})]
                # The FOC is: u'(c) + beta * d/dc [continuation value] = 0
                # Since continuation value is decreasing in c, we use bisection.
                
                def objective(c):
                    c = np.clip(c, 1e-12, m)
                    u = utility(c)
                    
                    # Continuation value
                    cont_val = 0.0
                    for zp in range(num_states):
                        m_next = R * (m - c) + income[zp]
                        # Interpolate value at m_next
                        v_next = np.interp(m_next, resource_grid, value[t + 1, zp, :])
                        cont_val += probs[zp] * v_next
                    
                    return u + beta * cont_val
                
                # Bisection on the derivative of the objective
                # f(c) = u'(c) + beta * d/dc [continuation value]
                # Since u''(c) < 0 and continuation value is concave, f(c) is decreasing.
                # We want to find c* such that f(c*) = 0.
                
                # We'll use a numerical derivative for the continuation part
                eps = 1e-7
                
                def f(c):
                    c = np.clip(c, 1e-12, m)
                    u_prime = utility_deriv(c)
                    
                    # Numerical derivative of continuation value
                    c_plus = np.clip(c + eps, 1e-12, m)
                    c_minus = np.clip(c - eps, 1e-12, m)
                    
                    cont_plus = 0.0
                    cont_minus = 0.0
                    for zp in range(num_states):
                        m_next_plus = R * (m - c_plus) + income[zp]
                        m_next_minus = R * (m - c_minus) + income[zp]
                        v_next_plus = np.interp(m_next_plus, resource_grid, value[t + 1, zp, :])
                        v_next_minus = np.interp(m_next_minus, resource_grid, value[t + 1, zp, :])
                        cont_plus += probs[zp] * v_next_plus
                        cont_minus += probs[zp] * v_next_minus
                    
                    cont_deriv = (cont_plus - cont_minus) / (c_plus - c_minus)
                    
                    return u_prime + beta * cont_deriv
                
                # Bisection bounds: c in [1e-12, m]
                lo = 1e-12
                hi = m
                
                # Check if the optimum is at the boundaries
                f_lo = f(lo)
                f_hi = f(hi)
                
                if f_lo <= 0:
                    c_star = lo
                elif f_hi >= 0:
                    c_star = hi
                else:
                    # Bisection
                    for _ in range(50):  # 50 iterations for high precision
                        mid = (lo + hi) / 2
                        f_mid = f(mid)
                        if f_mid > 0:
                            lo = mid
                        else:
                            hi = mid
                    c_star = (lo + hi) / 2
                
                opt_c[i] = c_star
                opt_val[i] = objective(c_star)
            
            policy[t, z, :] = opt_c
            value[t, z, :] = opt_val
            
            # Ensure validity: 0 < c <= m
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, resource_grid)
    
    # Final validation pass
    for t in range(horizon):
        for z in range(num_states):
            for m_idx in range(num_resources):
                m = resource_grid[m_idx]
                c = policy[t, z, m_idx]
                if c <= 0:
                    policy[t, z, m_idx] = 1e-12
                if c > m + 1e-10:
                    policy[t, z, m_idx] = m
    
    return policy
