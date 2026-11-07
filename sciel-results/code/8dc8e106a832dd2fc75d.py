import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s limit

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

    # Fine grid for the value function and policy
    n_fine = 801
    m_fine = np.geomspace(0.001, 40, n_fine)
    m_lo, m_hi = m_fine[0], m_fine[-1]

    V = np.zeros((horizon, Z, n_fine))
    c_policy = np.zeros((horizon, Z, n_fine))

    # Terminal: consume everything
    for z in range(Z):
        V[horizon - 1, z, :] = utility(m_fine)
        c_policy[horizon - 1, z, :] = m_fine.copy()

    # Helper: compute F(c) = u'(c) - beta*R*sum_z' P[z,z']*u'(c'(m',z'))
    # and its derivative for Newton. Vectorized over zp.
    def euler_F(m_val, c, z, t):
        # c_next for each zp: interp c_policy[t+1, zp, :] at m'
        m_prime = R * (m_val - c) + income  # shape (Z,)
        m_prime = np.clip(m_prime, m_lo, m_hi)
        c_next = np.array([np.interp(m_prime[i], m_fine, c_policy[t + 1, i, :]) for i in range(Z)])
        c_next = np.maximum(c_next, 1e-12)
        cont = np.dot(transition[z, :], utility_prime(c_next))
        return utility_prime(c) - beta * R * cont

    def euler_F_prime(m_val, c, z, t):
        # Numerical derivative of F w.r.t. c
        dc = max(c * 1e-6, 1e-10)
        c_p = min(c + dc, m_val * (1 - 1e-12))
        c_m = max(c - dc, 1e-12)
        return (euler_F(m_val, c_p, z, t) - euler_F(m_val, c_m, z, t)) / (c_p - c_m)

    # Work backwards
    for t in range(horizon - 2, -1, -1):
        for z in range(Z):
            c_t_z = np.zeros(n_fine)
            V_t_z = np.zeros(n_fine)

            for i in range(n_fine):
                m_val = m_fine[i]
                # Initial guess: proportional
                c = max(0.4 * m_val, 1e-8)
                c = min(c, m_val * 0.999)

                # Bisection bracket: F(c) decreasing in c
                # At c -> small, F > 0 (u'(c) large); at c -> m, F < 0
                c_lo = max(1e-12, m_val * 1e-4)
                c_hi = m_val * 0.999999

                # Newton with bisection fallback
                for it in range(25):
                    F = euler_F(m_val, c, z, t)
                    if abs(F) < 1e-10:
                        break
                    if F > 0:
                        c_lo = c
                    else:
                        c_hi = c

                    Fp = euler_F_prime(m_val, c, z, t)
                    if abs(Fp) > 1e-14:
                        c_new = c - F / Fp
                    else:
                        c_new = 0.5 * (c_lo + c_hi)

                    # Keep in bracket
                    if c_new <= c_lo or c_new >= c_hi:
                        c_new = 0.5 * (c_lo + c_hi)

                    if abs(c_new - c) < 1e-11 * max(1.0, c):
                        c = c_new
                        break
                    c = c_new

                c = np.clip(c, 1e-12, m_val)
                c_t_z[i] = c

                # Value
                m_prime = R * (m_val - c) + income
                m_prime = np.clip(m_prime, m_lo, m_hi)
                cont = 0.0
                for zp in range(Z):
                    cont += transition[z, zp] * np.interp(m_prime[zp], m_fine, V[t + 1, zp, :])
                V_t_z[i] = utility(c) + beta * cont

            c_policy[t, z, :] = c_t_z
            V[t, z, :] = V_t_z

        if time.time() > deadline - 30:
            break

    # Extract policy on the original resource_grid by interpolation
    consumption = np.zeros((horizon, Z, M))
    for z in range(Z):
        consumption[horizon - 1, z, :] = m_grid.copy()

    for t in range(horizon - 1):
        for z in range(Z):
            consumption[t, z, :] = np.interp(m_grid, m_fine, c_policy[t, z, :])

    # Refinement pass: local search to catch any boundary/local issues
    # Use the V function for continuation values
    n_c_local = 31
    c_fracs = np.linspace(0.001, 1.0, n_c_local)
    for t in range(horizon - 1):
        for z in range(Z):
            m_vals = m_grid[:, np.newaxis]
            c_vals = m_vals * c_fracs[np.newaxis, :]
            c_vals = np.maximum(c_vals, 1e-12)

            cont_val = np.zeros((M, n_c_local))
            for zp in range(Z):
                m_prime = R * (m_vals - c_vals) + income[zp]
                m_prime = np.clip(m_prime, m_lo, m_hi)
                v_interp = np.apply_along_axis(
                    lambda row: np.interp(row, m_fine, V[t + 1, zp, :]), 1, m_prime)
                cont_val += transition[z, zp] * v_interp

            obj = utility(c_vals) + beta * cont_val
            best_idx = obj.argmax(axis=1)
            c_refined = c_vals[np.arange(M), best_idx]

            # Only use refined if it's close to the Newton solution (sanity)
            # Blend: take the max of Newton and refined to avoid undersaving
            # Actually just use refined where it differs significantly
            consumption[t, z, :] = c_refined

        if time.time() > deadline - 12:
            break

    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid + 1e-10)

    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=m_grid, neginf=1e-12)

    return consumption
