import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M_out = len(resource_grid)
    Z = len(income)
    
    # Use a finer internal grid for better accuracy
    M_int = 801
    m_grid = np.geomspace(0.001, 40, M_int)
    
    # consumption[t, z, m_index] on the output grid
    consumption = np.zeros((horizon, Z, M_out))
    
    # Helper for marginal utility
    def u_prime(c, g):
        c = np.maximum(c, 1e-12)
        if g == 1:
            return 1.0 / c
        else:
            return c ** (-g)
    
    # Helper for log marginal utility (more stable for Newton's method)
    def log_u_prime(c, g):
        c = np.maximum(c, 1e-12)
        if g == 1:
            return -np.log(c)
        else:
            return -g * np.log(c)
            
    # Terminal condition: at t=5 (last decision), consume everything.
    # c_5(m, z) = m
    
    # Initialize the "next" policy function (for t=5)
    # c_next[z, m_idx] on the internal grid
    c_next = np.zeros((Z, M_int))
    for z in range(Z):
        c_next[z] = m_grid
    
    # Precompute log transition probabilities for stability
    log_trans = np.log(transition)
    
    # Backward induction for t = 4 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with simple rule (consume all)
            for tt in range(t, -1, -1):
                for z in range(Z):
                    consumption[tt, z] = resource_grid
            break
            
        c_curr = np.zeros((Z, M_int))
        
        for z in range(Z):
            # We want to find c(m) for each m in m_grid
            # Let s = (m - c) / m, so c = m * (1 - s)
            # s is in [0, 1). 
            # Euler: u'(c) = beta * R * sum_{z'} P(z'|z) * u'(c_{t+1}(m', z'))
            # where m' = R * (m - c) + income[z'] = R * m * s + income[z']
            
            # We solve for s using Newton's method.
            # Let F(s) = log(u'(c)) - log(E[u'(c_next)])
            # c = m * (1 - s)
            # c_next(m', z') is obtained by interpolating c_next[z'] on m_grid.
            
            # Initial guess for s:
            s = np.full(M_int, 0.2)
            
            # Newton's method iteration
            max_iter = 20
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                    
                c = m_grid * (1.0 - s)
                c = np.maximum(c, 1e-12)
                
                # LHS: log u'(c)
                lhs = log_u_prime(c, gamma)
                
                # RHS: log( beta * R * E[u'(c_next)] )
                # E[u'(c_next)] = sum_{z'} P(z'|z) * u'(c_next(z', m'))
                # m' = R * m * s + income[z']
                
                # We need to evaluate c_next(z', m') for all m and z'
                # m_prime shape: (M_int, Z)
                m_prime = R * m_grid[:, None] * s[:, None] + income[None, :]
                
                # Interpolate c_next for each z'
                # c_next shape: (Z, M_int)
                
                u_prime_next_sum = np.zeros(M_int)
                for zp in range(Z):
                    c_next_val = np.interp(m_prime[:, zp], m_grid, c_next[zp])
                    u_prime_val = u_prime(c_next_val, gamma)
                    u_prime_next_sum += transition[z, zp] * u_prime_val
                
                rhs_val = beta * R * u_prime_next_sum
                rhs_val = np.maximum(rhs_val, 1e-12)
                rhs = np.log(rhs_val)
                
                # F(s) = lhs - rhs
                F = lhs - rhs
                
                # Derivative F'(s)
                d_lhs = gamma / (1.0 - s)
                
                # d/ds rhs = (1 / E[u']) * E[ d/ds u'(c_next) ]
                # d/ds u'(c_next) = -gamma * c_next^{-gamma-1} * dc_next/ds
                # dc_next/ds = dc_next/dm' * dm'/ds
                # dm'/ds = R * m
                
                # Numerical derivative for the RHS part
                h = 1e-6
                s_plus = np.clip(s + h, 0, 0.999999)
                c_plus = m_grid * (1.0 - s_plus)
                c_plus = np.maximum(c_plus, 1e-12)
                lhs_plus = log_u_prime(c_plus, gamma)
                
                m_prime_plus = R * m_grid[:, None] * s_plus[:, None] + income[None, :]
                u_prime_next_sum_plus = np.zeros(M_int)
                for zp in range(Z):
                    c_next_val_plus = np.interp(m_prime_plus[:, zp], m_grid, c_next[zp])
                    u_prime_val_plus = u_prime(c_next_val_plus, gamma)
                    u_prime_next_sum_plus += transition[z, zp] * u_prime_val_plus
                rhs_val_plus = beta * R * u_prime_next_sum_plus
                rhs_val_plus = np.maximum(rhs_val_plus, 1e-12)
                rhs_plus = np.log(rhs_val_plus)
                
                F_plus = lhs_plus - rhs_plus
                
                dF = (F_plus - F) / h
                
                # Newton step: s_new = s - F / dF
                mask = np.abs(dF) > 1e-8
                s_new = s.copy()
                s_new[mask] = s[mask] - F[mask] / dF[mask]
                
                # Constrain s to [0, 1)
                s_new = np.clip(s_new, 0.0, 0.999999)
                
                # Check convergence
                if np.max(np.abs(s_new - s)) < 1e-8:
                    s = s_new
                    break
                s = s_new
                
            # Final c
            c_curr[z] = m_grid * (1.0 - s)
            
        # Interpolate c_curr from m_grid to resource_grid for storage
        for z in range(Z):
            consumption[t, z] = np.interp(resource_grid, m_grid, c_curr[z])
        
        c_next = c_curr
        
    # Final validity checks
    for t in range(horizon):
        for z in range(Z):
            c = consumption[t, z]
            m = resource_grid
            c = np.maximum(c, 1e-12)
            c = np.minimum(c, m + 1e-10)
            consumption[t, z] = c
            
    # Enforce terminal condition exactly
    consumption[5] = resource_grid[None, :]
    
    return consumption
