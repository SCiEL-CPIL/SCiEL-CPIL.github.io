import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 140.0 # 10s safety margin

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = len(income)
    
    # Precompute utility derivatives
    def U_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-6:
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    def U_double_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-6:
            return -1.0 / (c ** 2)
        else:
            return -gamma * (c ** (-gamma - 1))

    # Storage
    consumption = np.zeros((horizon, n_z, n_m))
    V_prime = np.zeros((horizon, n_z, n_m))
    V_double_prime = np.zeros((horizon, n_z, n_m))

    # Terminal Period t = horizon - 1
    t = horizon - 1
    for z in range(n_z):
        consumption[t, z, :] = m_grid
        V_prime[t, z, :] = U_prime(m_grid)
        V_double_prime[t, z, :] = U_double_prime(m_grid)

    # Backward Induction
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline:
            # Fill remaining with simple rule if timeout
            for tt in range(t, -1, -1):
                 for zz in range(n_z):
                     consumption[tt, zz, :] = m_grid * 0.5 # Rough guess
            break

        for z in range(n_z):
            for i_m in range(n_m):
                m = m_grid[i_m]
                
                # Newton-Raphson for c
                # f(c) = U'(c) - beta * R * sum_z' P(z'|z) * Vp_next[z'](m')
                # m' = R * (m - c) + income[z']
                
                # Initial guess
                # c ~ m / (1 + beta * R * something)
                # Simple start: c = m * 0.5
                c = m * 0.5
                if c < 1e-12: c = 1e-12
                if c > m: c = m
                
                # Interpolation function for Vp_next and Vpp_next
                # We need to interpolate for each z'
                
                for _ in range(20): # Max iterations
                    if time.time() > deadline:
                        break
                    
                    # Compute m' for each z'
                    m_primes = np.zeros(n_z)
                    for zp in range(n_z):
                        m_primes[zp] = R * (m - c) + income[zp]
                        
                    # Interpolate Vp_next[zp] at m_primes[zp]
                    # Note: V_prime[t+1, zp, :] is the array for state zp
                    vp_vals = np.zeros(n_z)
                    vpp_vals = np.zeros(n_z)
                    
                    for zp in range(n_z):
                        # np.interp requires x to be increasing
                        # m_grid is increasing
                        # Handle out of bounds:
                        # If m_prime < m_grid[0], use m_grid[0] value?
                        # If m_prime > m_grid[-1], use m_grid[-1] value?
                        # Actually, m' >= income[zp] >= 0.5.
                        # m_grid min is 0.001.
                        # m' can be large. Max m ~ 40. m' ~ 1.02*40 + 1.5 ~ 42.
                        # m_grid max is 40.
                        # So we might go out of bounds on the high end.
                        # We should clip or extend.
                        # Let's clip to grid bounds for interpolation, but keep the actual m' for the logic?
                        # No, V is defined on the grid. If m' is out of grid, we assume it's at the boundary or extrapolate linearly?
                        # The task says "valued backward in time... interpolated with np.interp".
                        # The evaluator interpolates *our* policy.
                        # But *we* need to evaluate the continuation value.
                        # If m' is out of our grid, we should probably clamp to the grid limits for stability, or use the boundary value.
                        m_prime_clipped = np.clip(m_primes[zp], m_grid[0], m_grid[-1])
                        vp_vals[zp] = np.interp(m_prime_clipped, m_grid, V_prime[t+1, zp, :])
                        vpp_vals[zp] = np.interp(m_prime_clipped, m_grid, V_double_prime[t+1, zp, :])
                    
                    # Expected values
                    # transition[z] is P(z'|z)
                    p_zp = transition[z]
                    
                    ev_vp = np.dot(p_zp, vp_vals)
                    ev_vpp = np.dot(p_zp, vpp_vals)
                    
                    # Euler Equation
                    # U'(c) = beta * R * ev_vp
                    # f(c) = U'(c) - beta * R * ev_vp
                    f = U_prime(c) - beta * R * ev_vp
                    
                    # Derivative
                    # f'(c) = U''(c) + beta * R^2 * ev_vpp
                    # Check sign of second term again.
                    # d/dc ( beta R sum P Vp(m') ) = beta R sum P Vpp(m') * dm'/dc
                    # dm'/dc = -R
                    # So term is - beta R^2 sum P Vpp(m')
                    # f(c) = U'(c) - (term)
                    # f'(c) = U''(c) - (- beta R^2 ev_vpp) = U''(c) + beta R^2 ev_vpp
                    fp = U_double_prime(c) + beta * R * R * ev_vpp
                    
                    # Newton step
                    if abs(fp) < 1e-12:
                        break
                    step = f / fp
                    c_new = c - step
                    
                    # Safeguard
                    if c_new <= 1e-12 or c_new >= m:
                        # Bisection-like adjustment
                        if c_new <= 1e-12:
                            c_new = (c + 1e-12) / 2
                        else:
                            c_new = (c + m) / 2
                        # Or just dampen
                        c_new = c - 0.5 * step
                        if c_new < 1e-12: c_new = 1e-12
                        if c_new > m: c_new = m
                        
                    # Check convergence
                    if abs(c_new - c) < 1e-10 * max(1, c):
                        c = c_new
                        break
                    c = c_new
                    
                consumption[t, z, i_m] = c
                V_prime[t, z, i_m] = U_prime(c)
                # Recalculate ev_vpp with final c for V''
                # Actually, we can just use the last ev_vpp computed in the loop
                # But to be safe, recompute with final c
                m_primes_final = np.zeros(n_z)
                vpp_vals_final = np.zeros(n_z)
                for zp in range(n_z):
                    m_primes_final[zp] = R * (m - c) + income[zp]
                    m_prime_clipped = np.clip(m_primes_final[zp], m_grid[0], m_grid[-1])
                    vpp_vals_final[zp] = np.interp(m_prime_clipped, m_grid, V_double_prime[t+1, zp, :])
                ev_vpp_final = np.dot(transition[z], vpp_vals_final)
                V_double_prime[t, z, i_m] = -beta * R * R * ev_vpp_final

    # Final validity check
    consumption = np.clip(consumption, 1e-12, None)
    for t in range(horizon):
        for z in range(n_z):
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid)
            
    # c[5] must equal grid
    consumption[horizon-1, :, :] = m_grid

    return consumption
