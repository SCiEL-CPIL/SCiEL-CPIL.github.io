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

    # Create a fine grid for m values in log-space for smoother interpolation
    n_fine = 1000
    m_fine = np.geomspace(0.001, 40, n_fine)
    log_m_fine = np.log(m_fine)
    
    # V[t][z] will be a 1D array of length n_fine
    V = np.zeros((horizon, Z, n_fine))
    
    # Terminal: V[horizon-1][z][i] = u(m_fine[i])
    for z in range(Z):
        V[horizon-1, z, :] = utility(m_fine)
    
    # c_policy[t][z][i] for the fine grid
    c_policy = np.zeros((horizon, Z, n_fine))
    
    # Terminal: consume all
    for z in range(Z):
        c_policy[horizon-1, z, :] = m_fine.copy()
    
    # Precompute log-space interpolation weights for each m_fine point
    # For each i, we need to interpolate V[t+1, z, :] at m_prime values
    # We'll use np.interp but in log-space for smoother results
    
    # For t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 15:
            break
        
        for z in range(Z):
            c_t_z = np.zeros(n_fine)
            V_t_z = np.zeros(n_fine)
            
            # Vectorized approach: solve Euler equation for all m_fine points
            # F(c) = u'(c) - beta * R * sum_{z'} P[z,z'] * u'(c_next(m', z')) = 0
            # where m' = R*(m-c) + income[z']
            
            # Initial guess: proportional consumption based on CRRA approximation
            # For CRRA with constant relative risk aversion, c/m is roughly constant
            # Use a heuristic: c = m / (1 + beta*R) as starting point
            c = m_fine / (1.0 + beta * R)
            c = np.clip(c, 1e-10, m_fine * 0.9999)
            
            # Bisection method for robustness
            # F(c) is decreasing in c (as c increases, u'(c) decreases, and m' decreases so c_next decreases, u'(c_next) increases)
            # Wait: as c increases, m' = R*(m-c)+income decreases, so c_next decreases, u'(c_next) increases
            # So the second term increases, meaning F(c) = u'(c) - beta*R*sum... could go either way
            # Let's use a hybrid: try secant first, fall back to bisection
            
            for i in range(n_fine):
                m_val = m_fine[i]
                
                # Initial guess
                c_lo = 1e-10
                c_hi = m_val * 0.9999
                c = 0.5 * m_val
                
                # Try secant method for a few iterations
                c_prev = c * 0.9
                F_prev = None
                
                for sec_iter in range(8):
                    # Compute F(c)
                    F = 0.0
                    for zp in range(Z):
                        m_prime = R * (m_val - c) + income[zp]
                        m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                        c_next = np.interp(m_prime, m_fine, c_policy[t+1, zp, :])
                        c_next = max(c_next, 1e-12)
                        F += transition[z, zp] * utility_prime(c_next)
                    F = utility_prime(c) - beta * R * F
                    
                    if sec_iter == 0:
                        # Compute F at c_prev
                        F_prev = 0.0
                        for zp in range(Z):
                            m_prime_prev = R * (m_val - c_prev) + income[zp]
                            m_prime_prev = np.clip(m_prime_prev, m_fine[0], m_fine[-1])
                            c_next_prev = np.interp(m_prime_prev, m_fine, c_policy[t+1, zp, :])
                            c_next_prev = max(c_next_prev, 1e-12)
                            F_prev += transition[z, zp] * utility_prime(c_next_prev)
                        F_prev = utility_prime(c_prev) - beta * R * F_prev
                    else:
                        F_prev = F_old
                    
                    # Secant step
                    if abs(F - F_prev) > 1e-14:
                        c_new = c - F * (c - c_prev) / (F - F_prev)
                    else:
                        c_new = c
                    
                    c_new = np.clip(c_new, c_lo, c_hi)
                    
                    c_old = c
                    F_old = F
                    c = c_new
                    c_prev = c_old
                    
                    if abs(F) < 1e-11:
                        break
                
                # If secant didn't converge well, use bisection to polish
                if abs(F) > 1e-8:
                    c_lo = 1e-10
                    c_hi = m_val * 0.9999
                    
                    # Find bracket
                    F_lo = 0.0
                    for zp in range(Z):
                        m_prime = R * (m_val - c_lo) + income[zp]
                        m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                        c_next = np.interp(m_prime, m_fine, c_policy[t+1, zp, :])
                        c_next = max(c_next, 1e-12)
                        F_lo += transition[z, zp] * utility_prime(c_next)
                    F_lo = utility_prime(c_lo) - beta * R * F_lo
                    
                    F_hi = 0.0
                    for zp in range(Z):
                        m_prime = R * (m_val - c_hi) + income[zp]
                        m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                        c_next = np.interp(m_prime, m_fine, c_policy[t+1, zp, :])
                        c_next = max(c_next, 1e-12)
                        F_hi += transition[z, zp] * utility_prime(c_next)
                    F_hi = utility_prime(c_hi) - beta * R * F_hi
                    
                    # Bisection
                    for bis_iter in range(30):
                        c_mid = 0.5 * (c_lo + c_hi)
                        F_mid = 0.0
                        for zp in range(Z):
                            m_prime = R * (m_val - c_mid) + income[zp]
                            m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                            c_next = np.interp(m_prime, m_fine, c_policy[t+1, zp, :])
                            c_next = max(c_next, 1e-12)
                            F_mid += transition[z, zp] * utility_prime(c_next)
                        F_mid = utility_prime(c_mid) - beta * R * F_mid
                        
                        if F_mid * F_lo < 0:
                            c_hi = c_mid
                            F_hi = F_mid
                        else:
                            c_lo = c_mid
                            F_lo = F_mid
                        
                        if abs(c_hi - c_lo) < 1e-12 * max(1.0, m_val):
                            break
                    
                    c = 0.5 * (c_lo + c_hi)
                
                c_t_z[i] = c
                
                # Compute V_t_z[i]
                cont_val = 0.0
                for zp in range(Z):
                    m_prime = R * (m_val - c) + income[zp]
                    m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                    cont_val += transition[z, zp] * np.interp(m_prime, m_fine, V[t+1, zp, :])
                V_t_z[i] = utility(c) + beta * cont_val
            
            c_policy[t, z, :] = c_t_z
            V[t, z, :] = V_t_z
    
    # Extract consumption policy on the original resource_grid
    consumption = np.zeros((horizon, Z, M))
    
    # Terminal period
    for z in range(Z):
        consumption[horizon-1, z, :] = m_grid.copy()
    
    # For t = 0 to horizon-2, interpolate from the fine grid policy
    for t in range(horizon - 1):
        for z in range(Z):
            consumption[t, z, :] = np.interp(m_grid, m_fine, c_policy[t, z, :])
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid + 1e-10)
    
    # Ensure finite values
    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=m_grid, neginf=1e-12)
    
    return consumption
