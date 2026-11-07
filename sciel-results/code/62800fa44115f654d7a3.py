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

    # Use a fine geometric grid for the value function
    fine_M = 801
    fine_grid = np.geomspace(0.001, 40, fine_M)

    # Precompute utility and marginal utility
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Backward induction
    # V[t, z, m_idx] = value at time t, income state z, resource level fine_grid[m_idx]
    V = np.zeros((horizon + 1, Z, fine_M))
    consumption_fine = np.zeros((horizon, Z, fine_M))

    # Terminal: at t = horizon, no more utility
    # V[horizon] = 0

    for t in range(horizon - 1, -1, -1):
        for z in range(Z):
            if t == horizon - 1:
                # Last period: must consume all resources
                consumption_fine[t, z, :] = fine_grid
                V[t, z, :] = utility(fine_grid)
                continue

            # For each m on the fine grid, find optimal c
            # Euler equation: u'(c) = beta * R * E[u'(c')]
            # where c' is next period's consumption given m' = R*(m-c) + income[z']
            
            # We'll use a combination of:
            # 1. Grid search to get a good initial guess
            # 2. Newton refinement on the Euler equation
            
            m_vals = fine_grid
            c_guess = np.zeros(fine_M)
            
            # Initial guess: use a coarse grid search
            # c ranges from a small fraction to m
            # For efficiency, use a moderate number of c-points
            n_c = 150
            c_fracs = np.linspace(0.005, 1.0, n_c)
            
            best_val = -np.inf * np.ones(fine_M)
            best_c = np.zeros(fine_M)
            
            # Vectorized grid search
            for c_frac in c_fracs:
                c_vals = m_vals * c_frac
                c_vals = np.minimum(c_vals, m_vals)  # c <= m
                c_vals = np.maximum(c_vals, 1e-12)
                
                m_minus_c = m_vals - c_vals
                m_minus_c = np.maximum(m_minus_c, 0)
                
                # Compute expected continuation value
                cont = np.zeros(fine_M)
                for zp in range(Z):
                    p = transition[z, zp]
                    if p < 1e-15:
                        continue
                    m_prime = R * m_minus_c + income[zp]
                    m_prime = np.maximum(m_prime, 1e-12)
                    # Clip to grid range
                    m_prime = np.clip(m_prime, fine_grid[0], fine_grid[-1])
                    v_next = np.interp(m_prime, fine_grid, V[t + 1, zp, :])
                    cont += p * v_next
                
                total = utility(c_vals) + beta * cont
                
                # Update best
                mask = total > best_val
                best_val[mask] = total[mask]
                best_c[mask] = c_vals[mask]
            
            c_guess = best_c
            consumption_fine[t, z, :] = c_guess
            
            # Newton refinement on the Euler equation
            # u'(c) - beta * R * E[u'(c')] = 0
            # where c' is determined by m' = R*(m-c) + income[z']
            # and c' is the optimal consumption at (t+1, z', m')
            
            # We need the derivative of the Euler residual with respect to c
            # This requires the derivative of c' with respect to m', which is dc'/dm'
            # For now, use a simple secant/Newton hybrid
            
            # Perform a few Newton iterations
            for newton_iter in range(3):
                if time.time() > deadline - 30.0:
                    break
                    
                # Compute Euler residual
                residual = np.zeros(fine_M)
                for i in range(fine_M):
                    c = c_guess[i]
                    m = m_vals[i]
                    
                    # u'(c)
                    mu_c = marginal_utility(c)
                    
                    # Expected beta*R*u'(c')
                    exp_mu_cp = 0.0
                    for zp in range(Z):
                        p = transition[z, zp]
                        if p < 1e-15:
                            continue
                        m_prime = R * (m - c) + income[zp]
                        m_prime = np.clip(m_prime, fine_grid[0], fine_grid[-1])
                        
                        # Get c' at (t+1, zp, m_prime)
                        c_prime = np.interp(m_prime, fine_grid, consumption_fine[t + 1, zp, :])
                        c_prime = np.maximum(c_prime, 1e-12)
                        
                        mu_cp = marginal_utility(c_prime)
                        exp_mu_cp += p * mu_cp
                    
                    residual[i] = mu_c - beta * R * exp_mu_cp
                
                # Check convergence
                if np.max(np.abs(residual)) < 1e-8:
                    break
                
                # Compute derivative of residual w.r.t. c (approximate)
                # d/dc [u'(c) - beta*R*E[u'(c')]]
                # = u''(c) - beta*R*E[u''(c') * dc'/dm' * dm'/dc]
                # dm'/dc = -R
                # dc'/dm' is the derivative of the optimal consumption policy
                # Approximate with a finite difference
                
                deriv = np.zeros(fine_M)
                h = 1e-5
                for i in range(fine_M):
                    c = c_guess[i]
                    m = m_vals[i]
                    
                    # u''(c)
                    if abs(gamma - 1.0) < 1e-10:
                        u2_c = -1.0 / (c ** 2)
                    else:
                        u2_c = -gamma * c ** (-gamma - 1)
                    
                    # Expected term
                    exp_term = 0.0
                    for zp in range(Z):
                        p = transition[z, zp]
                        if p < 1e-15:
                            continue
                        m_prime = R * (m - c) + income[zp]
                        m_prime = np.clip(m_prime, fine_grid[0], fine_grid[-1])
                        
                        # dc'/dm' via finite difference
                        m_prime_plus = np.clip(m_prime + h, fine_grid[0], fine_grid[-1])
                        m_prime_minus = np.clip(m_prime - h, fine_grid[0], fine_grid[-1])
                        c_prime_plus = np.interp(m_prime_plus, fine_grid, consumption_fine[t + 1, zp, :])
                        c_prime_minus = np.interp(m_prime_minus, fine_grid, consumption_fine[t + 1, zp, :])
                        dc_dm = (c_prime_plus - c_prime_minus) / (m_prime_plus - m_prime_minus + 1e-15)
                        
                        c_prime = np.interp(m_prime, fine_grid, consumption_fine[t + 1, zp, :])
                        c_prime = np.maximum(c_prime, 1e-12)
                        
                        # u''(c')
                        if abs(gamma - 1.0) < 1e-10:
                            u2_cp = -1.0 / (c_prime ** 2)
                        else:
                            u2_cp = -gamma * c_prime ** (-gamma - 1)
                        
                        exp_term += p * u2_cp * dc_dm * (-R)
                    
                    deriv[i] = u2_c - beta * R * exp_term
                
                # Newton step
                # c_new = c - residual / deriv
                # But we need to ensure 0 < c <= m
                for i in range(fine_M):
                    if abs(deriv[i]) > 1e-12:
                        c_new = c_guess[i] - residual[i] / deriv[i]
                        c_guess[i] = np.clip(c_new, 1e-12, m_vals[i])
            
            consumption_fine[t, z, :] = c_guess
            
            # Compute V[t, z, :] using the optimal c
            c_vals = c_guess
            m_minus_c = np.maximum(m_vals - c_vals, 0)
            cont = np.zeros(fine_M)
            for zp in range(Z):
                p = transition[z, zp]
                if p < 1e-15:
                    continue
                m_prime = R * m_minus_c + income[zp]
                m_prime = np.clip(m_prime, fine_grid[0], fine_grid[-1])
                v_next = np.interp(m_prime, fine_grid, V[t + 1, zp, :])
                cont += p * v_next
            V[t, z, :] = utility(c_vals) + beta * cont
        
        # Check deadline
        if time.time() > deadline - 20.0:
            break

    # Extract policy on original resource_grid
    consumption = np.zeros((horizon, Z, M))
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.interp(resource_grid, fine_grid, consumption_fine[t, z, :])

    # Enforce terminal condition: c[5] = m
    consumption[horizon - 1, :, :] = resource_grid[None, None, :]

    # Enforce 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            m_vals = resource_grid[None, :]
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_vals + 1e-10)

    return consumption
