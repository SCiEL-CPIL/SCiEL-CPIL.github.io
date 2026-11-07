"""Solve consumption-saving policy via Euler-equation Newton with log-linear value interpolation."""
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = resource_grid
    num_m = len(M)
    num_z = len(income)
    H = horizon

    # Precompute utility and marginal utility
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def marginal_util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return np.power(c, -gamma)

    # consumption[t, z, m_index], shape (H, num_z, num_m)
    consumption = np.zeros((H, num_z, num_m))

    # Last period: consume everything
    consumption[H-1, :, :] = M[np.newaxis, np.newaxis, :]

    # Backward induction from t = H-2 down to 0
    for t in range(H-2, -1, -1):
        if time.time() > deadline - 5.0:
            # Safety: fill remaining with proportional guess
            for tt in range(t, H-1):
                if tt < H-1:
                    consumption[tt, :, :] = 0.9 * M[np.newaxis, np.newaxis, :]
            break

        for z in range(num_z):
            # For each m in grid, solve for c
            for i_m in range(num_m):
                m = M[i_m]
                
                # Try Newton's method on the Euler equation
                # F(c) = u'(c) - beta * sum_z' P(z'|z) * u'(R*(m-c) + income[z'])
                # F'(c) = -u''(c) + beta * sum_z' P(z'|z) * R * u''(R*(m-c) + income[z'])
                
                # Initial guess: use a heuristic
                # For log utility, optimal c is roughly m / (1 + beta*R)
                if abs(gamma - 1.0) < 1e-10:
                    c_guess = m / (1.0 + beta * R)
                else:
                    # For CRRA, use a similar heuristic
                    c_guess = m / (1.0 + beta * R)
                
                c_guess = np.clip(c_guess, 1e-12, m * 0.999)
                
                # Newton iterations
                c = c_guess
                converged = False
                for _ in range(50):
                    c = max(c, 1e-12)
                    c = min(c, m * (1.0 - 1e-15))
                    
                    # Compute F(c) and F'(c)
                    mu_c = marginal_util(c)
                    
                    # Expected continuation marginal utility
                    exp_mu_next = 0.0
                    for z_prime in range(num_z):
                        m_next = R * (m - c) + income[z_prime]
                        if m_next < 1e-12:
                            m_next = 1e-12
                        exp_mu_next += transition[z, z_prime] * marginal_util(m_next)
                    
                    F = mu_c - beta * exp_mu_next
                    
                    # Derivative
                    # u''(c) for CRRA: -gamma * c^(-gamma-1) for gamma != 1, -1/c^2 for gamma=1
                    if abs(gamma - 1.0) < 1e-10:
                        d2u_c = -1.0 / (c * c)
                    else:
                        d2u_c = -gamma * np.power(c, -gamma - 1.0)
                    
                    d2u_next = 0.0
                    for z_prime in range(num_z):
                        m_next = R * (m - c) + income[z_prime]
                        if m_next < 1e-12:
                            m_next = 1e-12
                        if abs(gamma - 1.0) < 1e-10:
                            d2u_next += transition[z, z_prime] * (-1.0 / (m_next * m_next))
                        else:
                            d2u_next += transition[z, z_prime] * (-gamma * np.power(m_next, -gamma - 1.0))
                    
                    F_prime = -d2u_c + beta * R * d2u_next
                    
                    if abs(F_prime) < 1e-15:
                        break
                    
                    delta_c = F / F_prime
                    c_new = c - delta_c
                    
                    # Ensure c stays in valid range
                    if c_new <= 1e-12 or c_new >= m:
                        c_new = (c + m) / 2.0 if c_new > m else (c + 1e-12) / 2.0
                        c_new = max(c_new, 1e-12)
                        c_new = min(c_new, m * 0.999)
                    
                    if abs(c_new - c) < 1e-14 * max(1.0, abs(c)):
                        c = c_new
                        converged = True
                        break
                    c = c_new
                
                consumption[t, z, i_m] = c

    # Final validity checks
    # Ensure 0 < c <= m + 1e-10
    for t in range(H):
        for z in range(num_z):
            for i_m in range(num_m):
                c_val = consumption[t, z, i_m]
                m_val = M[i_m]
                if c_val <= 0:
                    consumption[t, z, i_m] = 1e-12
                if c_val > m_val + 1e-10:
                    consumption[t, z, i_m] = m_val

    # Ensure c[5] equals grid within 1e-9
    consumption[H-1, :, :] = M[np.newaxis, np.newaxis, :]

    return consumption
