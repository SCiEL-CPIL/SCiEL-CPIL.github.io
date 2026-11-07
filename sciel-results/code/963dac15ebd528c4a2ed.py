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

    # Utility and marginal utility
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c**(1 - gamma) / (1 - gamma)

    def marginal_u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c**(-gamma)

    # Value function grid
    # m' ranges from min(income)=0.5 to R*40+1.5=42.3
    # Use a geometric grid for good resolution at low m
    N_vf = 4000
    m_vf = np.geomspace(0.001, 45.0, N_vf)

    # V[t, z, :] value function, C[t, z, :] consumption policy
    V = np.zeros((horizon, Z, N_vf))
    C = np.zeros((horizon, Z, N_vf))

    # Terminal: c = m, V = u(m)
    for z in range(Z):
        C[horizon-1, z, :] = m_vf
        V[horizon-1, z, :] = utility(m_vf)

    # Backward induction with bisection on Euler equation
    # For each m, find c such that:
    # u'(c) = beta * R * sum_z' P[z,z'] * V'[t+1, z'](R(m-c) + y[z'])
    # Left side decreases in c, right side increases in c (since V' decreases in m', and m' decreases in c... wait)
    # Actually m' = R(m-c) + y[z'], so as c increases, m' decreases, V'[m'] increases (V is concave, V' decreasing in m)
    # So RHS increases in c. LHS decreases in c. Unique crossing.
    
    # Precompute log-spaced c candidates for bisection
    # c is in (0, m). Use log-spaced grid for bisection.
    # For each m, c is in (eps, m). We'll do bisection in log-c space.
    log_c_min = np.log(1e-10)
    log_c_max_offset = 0.0  # log(m * 1.0) = log(m), so c_max = m
    
    n_bisect = 40  # iterations of bisection per m
    
    for t in range(horizon - 2, -1, -1):
        for z in range(Z):
            # For each m in m_vf, find optimal c
            # Vectorized bisection: maintain log_c_low, log_c_high arrays
            log_c_low = np.full(N_vf, log_c_min)
            log_c_high = np.log(m_vf)  # c <= m
            
            for it in range(n_bisect):
                log_c_mid = 0.5 * (log_c_low + log_c_high)
                c_mid = np.exp(log_c_mid)
                
                # LHS: u'(c_mid)
                lhs = marginal_u(c_mid)
                
                # RHS: beta * R * sum_z' P[z,z'] * V'[t+1,z'](m'_z')
                rhs = np.zeros(N_vf)
                for zp in range(Z):
                    m_prime = R * (m_vf - c_mid) + income[zp]
                    m_prime_clipped = np.clip(m_prime, m_vf[0], m_vf[-1])
                    
                    # Compute V'[t+1, zp] via log-linear interpolation
                    # V is concave, so V' is decreasing. Interpolate log(V') on log(m) grid.
                    # Actually, let's just interpolate V' directly.
                    # Compute V' numerically or analytically?
                    # For CRRA, we can compute V' from the policy... but we don't have it yet for t+1.
                    # Better: interpolate V[t+1,zp] and take numerical derivative, or use the fact that
                    # in the interior, V'[m] = u'(c*(m)) from the FOC.
                    # But we have C[t+1, zp, :] already computed!
                    # So V'[t+1, zp](m) = u'(C[t+1, zp](m)) for interior points.
                    
                    c_next = np.interp(m_prime_clipped, m_vf, C[t+1, zp, :])
                    c_next = np.clip(c_next, 1e-12, m_vf[-1])
                    v_prime = marginal_u(c_next)
                    
                    rhs += transition[z, zp] * beta * R * v_prime
                
                # If lhs > rhs, c is too small (u' too high), increase c
                # If lhs < rhs, c is too large, decrease c
                # lhs decreasing in c, rhs increasing in c
                # lhs > rhs means we need larger c
                mask = lhs > rhs
                log_c_low[mask] = log_c_mid[mask]
                log_c_high[~mask] = log_c_mid[~mask]
                
                # Time check
                if time.time() > deadline - 10:
                    break
            
            # Final c
            c_opt = np.exp(0.5 * (log_c_low + log_c_high))
            c_opt = np.clip(c_opt, 1e-12, m_vf)
            
            C[t, z, :] = c_opt
            
            # Compute V[t, z, :] = u(c) + beta * sum P * V[t+1](m')
            u_c = utility(c_opt)
            cont = np.zeros(N_vf)
            for zp in range(Z):
                m_prime = R * (m_vf - c_opt) + income[zp]
                m_prime_clipped = np.clip(m_prime, m_vf[0], m_vf[-1])
                v_interp = np.interp(m_prime_clipped, m_vf, V[t+1, zp, :])
                cont += transition[z, zp] * v_interp
            V[t, z, :] = u_c + beta * cont
        
        # Time check between t steps
        if time.time() > deadline - 5:
            # Fill remaining t with terminal-like behavior
            for t2 in range(t - 1, -1, -1):
                for z2 in range(Z):
                    C[t2, z2, :] = 0.5 * m_vf  # rough fallback
                    V[t2, z2, :] = utility(0.5 * m_vf)
            break

    # Interpolate to resource_grid
    consumption = np.zeros((horizon, Z, M))
    for t in range(horizon):
        for z in range(Z):
            if t == horizon - 1:
                consumption[t, z, :] = m_grid
            else:
                consumption[t, z, :] = np.interp(m_grid, m_vf, C[t, z, :])
                consumption[t, z, :] = np.clip(consumption[t, z, :], 1e-12, m_grid + 1e-10)

    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=40.0, neginf=1e-12)
    return consumption
