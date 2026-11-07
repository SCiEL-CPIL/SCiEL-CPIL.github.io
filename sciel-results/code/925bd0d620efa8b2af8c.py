import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M = len(resource_grid)
    Z = len(income)
    
    # consumption[t, z, m_index]
    consumption = np.zeros((horizon, Z, M))
    
    # Terminal condition: c_H(m) = m
    consumption[horizon - 1] = resource_grid[np.newaxis, :]  # shape (Z, M)
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
    
    def utility_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # Work backwards from t = horizon-2 to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break
        
        for z in range(Z):
            # For each m in the grid, find optimal c
            for i_m in range(M):
                m = resource_grid[i_m]
                
                # Try to find c in (0, m) that satisfies Euler equation
                # u'(c) = beta * sum_{z'} P(z'|z) * V'_{t+1}(R(m-c) + income[z'], z') * R
                
                # V'_{t+1} is the derivative of the value function w.r.t. m'
                # V_{t+1}(m', z') = u(c_{t+1}(m', z')) + beta * E[V_{t+2}]
                # But we can approximate V'_{t+1} using the policy and chain rule
                # V'_{t+1}(m', z') = u'(c_{t+1}(m', z')) * dc_{t+1}/dm'
                # Actually, by the envelope theorem: V'_{t+1}(m', z') = u'(c_{t+1}(m', z')) * (1 - R * dc_{t+1}/dm')... 
                
                # Simpler: use the fact that at optimum, V'(m') = u'(c(m'))
                # So V'_{t+1}(m', z') = u'(c_{t+1}(m', z'))
                
                # So Euler: u'(c) = beta * R * sum_{z'} P(z'|z) * u'(c_{t+1}(R(m-c) + income[z'], z'))
                
                # Let's define f(c) = u'(c) - beta * R * sum_{z'} P(z'|z) * u'(c_{t+1}(R(m-c) + income[z'], z'))
                # We want f(c) = 0
                
                # c must be in (0, m)
                c_lo = 1e-12
                c_hi = m - 1e-12
                
                # Check if solution exists in range
                def euler_residual(c):
                    m_prime_list = []
                    for z_prime in range(Z):
                        m_p = R * (m - c) + income[z_prime]
                        # Clamp m_p to grid range for interpolation
                        m_p_clamped = np.clip(m_p, resource_grid[0], resource_grid[-1])
                        # Interpolate c_{t+1} at m_p_clamped for state z_prime
                        c_next = np.interp(m_p_clamped, resource_grid, consumption[t + 1, z_prime, :])
                        m_prime_list.append(c_next)
                    
                    expected_mu = 0.0
                    for z_prime in range(Z):
                        expected_mu += transition[z, z_prime] * utility_prime(m_prime_list[z_prime])
                    
                    return utility_prime(c) - beta * R * expected_mu
                
                # Bisection
                f_lo = euler_residual(c_lo)
                f_hi = euler_residual(c_hi)
                
                # If f_lo and f_hi have same sign, the optimal is at boundary
                if f_lo * f_hi > 0:
                    # Check which boundary is better
                    # If f_lo < 0, we want higher c (since u' is decreasing, f increases with c)
                    # Actually u' is decreasing, so as c increases, u'(c) decreases
                    # f(c) = u'(c) - beta*R*E[u'(c')]
                    # As c increases, u'(c) decreases, and m' = R(m-c)+income decreases, so c' decreases, u'(c') increases
                    # So f(c) is decreasing in c
                    # If f_lo < 0 and f_hi < 0, then f < 0 everywhere, optimal at c_hi
                    # If f_lo > 0 and f_hi > 0, then f > 0 everywhere, optimal at c_lo
                    if f_lo < 0:
                        c_opt = c_hi
                    else:
                        c_opt = c_lo
                else:
                    # Bisection
                    for _ in range(50):
                        c_mid = (c_lo + c_hi) / 2
                        f_mid = euler_residual(c_mid)
                        if abs(f_mid) < 1e-12:
                            break
                        if f_lo * f_mid <= 0:
                            c_hi = c_mid
                            f_hi = f_mid
                        else:
                            c_lo = c_mid
                            f_lo = f_mid
                        if time.time() > deadline:
                            break
                    c_opt = (c_lo + c_hi) / 2
                
                consumption[t, z, i_m] = c_opt
    
    # Ensure c[5] equals grid
    consumption[horizon - 1] = resource_grid[np.newaxis, :]
    
    # Validate: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            for i_m in range(M):
                m = resource_grid[i_m]
                c = consumption[t, z, i_m]
                if c <= 0:
                    consumption[t, z, i_m] = 1e-12
                if c > m + 1e-10:
                    consumption[t, z, i_m] = m
    
    return consumption
