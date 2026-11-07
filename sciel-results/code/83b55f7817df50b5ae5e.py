import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = len(resource_grid)
    Z = len(income)
    m_grid = resource_grid

    # CRRA utility and derivatives
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            c = np.maximum(c, 1e-12)
            return np.log(c)
        def utility_prime(c):
            c = np.maximum(c, 1e-12)
            return 1.0 / c
    else:
        def utility(c):
            c = np.maximum(c, 1e-12)
            return c**(1 - gamma) / (1 - gamma)
        def utility_prime(c):
            c = np.maximum(c, 1e-12)
            return c**(-gamma)

    # Use a fine grid for m values
    m_fine = np.geomspace(0.001, 40, 1201)
    n_fine = len(m_fine)
    m_lo = m_fine[0]
    m_hi = m_fine[-1]
    
    # V[t][z] will be a 1D array of length n_fine
    V = np.zeros((horizon, Z, n_fine))
    
    # Terminal: V[horizon-1][z][i] = u(m_fine[i])
    for z in range(Z):
        V[horizon-1, z, :] = utility(m_fine)
    
    # c_policy[t][z][i] stores the optimal consumption at m_fine[i] for state z at time t
    c_policy = np.zeros((horizon, Z, n_fine))
    
    # Terminal: consume all
    for z in range(Z):
        c_policy[horizon-1, z, :] = m_fine.copy()
    
    # Work backwards from t = horizon-2 down to 0, fully vectorized over m
    for t in range(horizon - 2, -1, -1):
        for z in range(Z):
            c = np.clip(0.5 * m_fine, 1e-8, m_fine * 0.999)
            c_lo = 1e-12
            c_hi = m_fine * (1.0 - 1e-10)
            
            for newton_iter in range(25):
                # F(c) = u'(c) - beta*R * sum_{z'} P[z,z'] * u'(c_policy(t+1,z', m'))
                F = utility_prime(c)
                for zp in range(Z):
                    m_prime = R * (m_fine - c) + income[zp]
                    m_prime = np.clip(m_prime, m_lo, m_hi)
                    c_next = np.interp(m_prime, m_fine, c_policy[t+1, zp, :])
                    c_next = np.maximum(c_next, 1e-12)
                    F -= beta * R * transition[z, zp] * utility_prime(c_next)
                
                # Numerical derivative F'(c)
                dc = np.maximum(c * 1e-6, 1e-12)
                c_plus = np.minimum(c + dc, c_hi)
                c_minus = np.maximum(c - dc, c_lo)
                actual_dc = c_plus - c_minus
                
                F_plus = utility_prime(c_plus)
                for zp in range(Z):
                    mp = R * (m_fine - c_plus) + income[zp]
                    mp = np.clip(mp, m_lo, m_hi)
                    cn = np.interp(mp, m_fine, c_policy[t+1, zp, :])
                    cn = np.maximum(cn, 1e-12)
                    F_plus -= beta * R * transition[z, zp] * utility_prime(cn)
                
                F_minus = utility_prime(c_minus)
                for zp in range(Z):
                    mp = R * (m_fine - c_minus) + income[zp]
                    mp = np.clip(mp, m_lo, m_hi)
                    cn = np.interp(mp, m_fine, c_policy[t+1, zp, :])
                    cn = np.maximum(cn, 1e-12)
                    F_minus -= beta * R * transition[z, zp] * utility_prime(cn)
                
                F_prime = (F_plus - F_minus) / actual_dc
                
                # Newton step
                c_new = c - F / F_prime
                c_new = np.clip(c_new, c_lo, c_hi)
                
                if np.all(np.abs(c_new - c) < 1e-13):
                    c = c_new
                    break
                c = c_new
            
            c_policy[t, z, :] = c
            
            # Compute V[t,z]
            V_t_z = utility(c)
            for zp in range(Z):
                m_prime = R * (m_fine - c) + income[zp]
                m_prime = np.clip(m_prime, m_lo, m_hi)
                V_t_z += beta * transition[z, zp] * np.interp(m_prime, m_fine, V[t+1, zp, :])
            V[t, z, :] = V_t_z
        
        # Check time
        if time.time() > deadline - 30:
            break
    
    # Extract consumption policy on the original resource_grid
    consumption = np.zeros((horizon, Z, M))
    
    # Terminal period
    for z in range(Z):
        consumption[horizon-1, z, :] = m_grid.copy()
    
    # For t = 0 to horizon-2, interpolate from the fine grid policy
    for t in range(horizon - 1):
        for z in range(Z):
            consumption[t, z, :] = np.interp(m_grid, m_fine, c_policy[t, z, :])
    
    # Final polish: refine on the output grid using the interpolated policy as initial guess
    if time.time() < deadline - 10:
        for t in range(horizon - 2, -1, -1):
            for z in range(Z):
                c = consumption[t, z, :].copy()
                c = np.clip(c, 1e-12, m_grid * (1.0 - 1e-10))
                
                for newton_iter in range(15):
                    F = utility_prime(c)
                    for zp in range(Z):
                        m_prime = R * (m_grid - c) + income[zp]
                        m_prime = np.clip(m_prime, m_lo, m_hi)
                        c_next = np.interp(m_prime, m_fine, c_policy[t+1, zp, :])
                        c_next = np.maximum(c_next, 1e-12)
                        F -= beta * R * transition[z, zp] * utility_prime(c_next)
                    
                    dc = np.maximum(c * 1e-6, 1e-12)
                    c_plus = np.minimum(c + dc, m_grid * (1.0 - 1e-10))
                    c_minus = np.maximum(c - dc, 1e-12)
                    actual_dc = c_plus - c_minus
                    
                    F_plus = utility_prime(c_plus)
                    for zp in range(Z):
                        mp = R * (m_grid - c_plus) + income[zp]
                        mp = np.clip(mp, m_lo, m_hi)
                        cn = np.interp(mp, m_fine, c_policy[t+1, zp, :])
                        cn = np.maximum(cn, 1e-12)
                        F_plus -= beta * R * transition[z, zp] * utility_prime(cn)
                    
                    F_minus = utility_prime(c_minus)
                    for zp in range(Z):
                        mp = R * (m_grid - c_minus) + income[zp]
                        mp = np.clip(mp, m_lo, m_hi)
                        cn = np.interp(mp, m_fine, c_policy[t+1, zp, :])
                        cn = np.maximum(cn, 1e-12)
                        F_minus -= beta * R * transition[z, zp] * utility_prime(cn)
                    
                    F_prime = (F_plus - F_minus) / actual_dc
                    c_new = c - F / F_prime
                    c_new = np.clip(c_new, 1e-12, m_grid * (1.0 - 1e-10))
                    
                    if np.all(np.abs(c_new - c) < 1e-13):
                        c = c_new
                        break
                    c = c_new
                
                consumption[t, z, :] = c
        
        # Terminal is already correct
        for z in range(Z):
            consumption[horizon-1, z, :] = m_grid.copy()
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid + 1e-10)
    
    # Ensure finite values
    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=m_grid, neginf=1e-12)
    
    return consumption
