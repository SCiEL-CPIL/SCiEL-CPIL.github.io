import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    # Use a finer internal grid for better interpolation of the policy function
    # The resource_grid is 101 points. We use 201 points internally.
    # Extend range slightly to handle m' > 40
    m_internal = np.geomspace(0.001, 50.0, 201)
    M_int = len(m_internal)
    Z = len(income)
    
    # Precompute utility derivatives
    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c**(-gamma)

    def u_prime_scalar(c):
        c = max(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c**(-gamma)

    # policies[t] will be a 2D array of shape (Z, M_int)
    # policies[t][z] is the consumption policy for state z at time t, evaluated on m_internal
    policies = [None] * horizon

    # Terminal period: c = m
    for z in range(Z):
        policies[horizon - 1] = np.tile(m_internal, (Z, 1))
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        c_curr = np.zeros((Z, M_int))
        c_next = policies[t + 1]
        
        for z in range(Z):
            for i in range(M_int):
                m = m_internal[i]
                
                # Function to evaluate the RHS of the Euler equation
                # RHS(c) = beta * R * sum_{z'} P[z, z'] * u'(c_{t+1}(R(m-c) + y_{z'}, z'))
                def euler_rhs(c_val):
                    # c_val is scalar
                    m_prime_vals = R * (m - c_val) + income
                    # Clip m_prime to internal grid range for interpolation
                    # np.interp handles extrapolation by returning edge values if not specified, 
                    # but default is to return y[0] or y[-1]? 
                    # Actually, np.interp(x, xp, fp) returns fp[0] if x < xp[0] and fp[-1] if x > xp[-1].
                    # This is a reasonable approximation (clamping).
                    c_next_vals = np.interp(m_prime_vals, m_internal, c_next)
                    u_prime_next = u_prime(c_next_vals)
                    expected_u_prime = np.dot(transition[z], u_prime_next)
                    return beta * R * expected_u_prime

                # We want to find c such that u'(c) - euler_rhs(c) = 0
                # Let f(c) = u'(c) - euler_rhs(c)
                # f(c) is decreasing in c (u' is decreasing, euler_rhs is increasing in c? 
                # m' decreases as c increases. c_{t+1} decreases. u'(c_{t+1}) increases.
                # So euler_rhs increases. Thus f(c) is strictly decreasing.)
                
                c_low = 1e-12
                c_high = m
                
                f_low = u_prime_scalar(c_low) - euler_rhs(c_low)
                f_high = u_prime_scalar(c_high) - euler_rhs(c_high)
                
                if f_low < 0:
                    c_opt = c_low
                elif f_high > 0:
                    c_opt = c_high
                else:
                    # Bisection
                    for _ in range(50):
                        c_mid = (c_low + c_high) / 2.0
                        f_mid = u_prime_scalar(c_mid) - euler_rhs(c_mid)
                        if f_mid > 0:
                            c_low = c_mid
                        else:
                            c_high = c_mid
                        if c_high - c_low < 1e-10:
                            break
                    c_opt = (c_low + c_high) / 2.0
                
                c_curr[z, i] = c_opt
        
        policies[t] = c_curr
        
        # Time check
        if time.time() > deadline - 5:
            break

    # Construct the output array
    M_out = len(resource_grid)
    consumption = np.zeros((horizon, Z, M_out))
    
    for t in range(horizon):
        for z in range(Z):
            # Interpolate from internal grid to output grid
            consumption[t, z, :] = np.interp(resource_grid, m_internal, policies[t][z, :])
            
            # Enforce constraints
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], resource_grid + 1e-10)
            
            # Terminal condition check
            if t == horizon - 1:
                consumption[t, z, :] = resource_grid.copy()

    # Final validity check
    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=40.0, neginf=1e-12)
    
    return consumption
