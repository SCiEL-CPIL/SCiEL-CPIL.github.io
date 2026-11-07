import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = int(model['horizon'])

    M = len(resource_grid)
    Z = len(income)

    # Fine grid for value function
    fine_M = 1501
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

    # Second derivative of utility
    def utility_second(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / (c ** 2)
        else:
            return -gamma * c ** (-gamma - 1)

    # Backward induction
    V = np.zeros((horizon + 1, Z, fine_M))
    consumption_fine = np.zeros((horizon, Z, fine_M))

    for t in range(horizon - 1, -1, -1):
        for z in range(Z):
            if t == horizon - 1:
                consumption_fine[t, z, :] = fine_grid
                V[t, z, :] = utility(fine_grid)
                continue

            m_vals = fine_grid
            c_guess = np.zeros(fine_M)

            # Initial guess: coarse grid search with more points
            n_c = 300
            c_fracs = np.linspace(0.002, 1.0, n_c)

            best_val = -np.inf * np.ones(fine_M)
            best_c = np.zeros(fine_M)

            for c_frac in c_fracs:
                c_vals = m_vals * c_frac
                c_vals = np.minimum(c_vals, m_vals)
                c_vals = np.maximum(c_vals, 1e-12)

                m_minus_c = m_vals - c_vals
                m_minus_c = np.maximum(m_minus_c, 0)

                cont = np.zeros(fine_M)
                for zp in range(Z):
                    p = transition[z, zp]
                    if p < 1e-15:
                        continue
                    m_prime = R * m_minus_c + income[zp]
                    m_prime = np.maximum(m_prime, 1e-12)
                    m_prime = np.clip(m_prime, fine_grid[0], fine_grid[-1])
                    v_next = np.interp(m_prime, fine_grid, V[t + 1, zp, :])
                    cont += p * v_next

                total = utility(c_vals) + beta * cont

                mask = total > best_val
                best_val[mask] = total[mask]
                best_c[mask] = c_vals[mask]

            c_guess = best_c
            consumption_fine[t, z, :] = c_guess

            # Newton refinement on the Euler equation
            # Euler: u'(c) = beta * R * E[u'(c') * c'(m')]
            # where c'(m') = dc/dm evaluated at m'
            for newton_iter in range(8):
                if time.time() > deadline - 25.0:
                    break

                c = c_guess.copy()
                residual = np.zeros(fine_M)
                deriv = np.zeros(fine_M)

                # Compute E[u'(c')] and its derivative w.r.t. m
                # We need: d/dm [ E[u'(c'(R(m-c)+y'))] ]
                # = E[u''(c') * c'(m') * R]
                # And the Euler residual is: u'(c) - beta*R*E[u'(c')]
                # Its derivative w.r.t. c is: u''(c) - beta*R*E[u''(c')*c'(m')*(-R)]
                # = u''(c) + beta*R^2*E[u''(c')*c'(m')]

                for i in range(fine_M):
                    m = m_vals[i]
                    ci = c[i]
                    mu_c = marginal_utility(ci)

                    exp_mu_cp = 0.0
                    exp_u2_cp_dc_dm = 0.0

                    for zp in range(Z):
                        p = transition[z, zp]
                        if p < 1e-15:
                            continue
                        m_prime = R * (m - ci) + income[zp]
                        m_prime = np.clip(m_prime, fine_grid[0], fine_grid[-1])

                        c_prime = np.interp(m_prime, fine_grid, consumption_fine[t + 1, zp, :])
                        c_prime = np.maximum(c_prime, 1e-12)

                        mu_cp = marginal_utility(c_prime)
                        u2_cp = utility_second(c_prime)

                        # Estimate c'(m') = dc/dm at m_prime using central difference
                        h = max(fine_grid[1] - fine_grid[0], 1e-5)
                        m_prime_plus = np.clip(m_prime + h, fine_grid[0], fine_grid[-1])
                        m_prime_minus = np.clip(m_prime - h, fine_grid[0], fine_grid[-1])
                        c_prime_plus = np.interp(m_prime_plus, fine_grid, consumption_fine[t + 1, zp, :])
                        c_prime_minus = np.interp(m_prime_minus, fine_grid, consumption_fine[t + 1, zp, :])
                        dc_dm = (c_prime_plus - c_prime_minus) / (m_prime_plus - m_prime_minus + 1e-15)

                        exp_mu_cp += p * mu_cp
                        exp_u2_cp_dc_dm += p * u2_cp * dc_dm

                    residual[i] = mu_c - beta * R * exp_mu_cp
                    deriv[i] = utility_second(ci) + beta * R * R * exp_u2_cp_dc_dm

                # Check convergence
                max_res = np.max(np.abs(residual))
                if max_res < 1e-10:
                    break

                # Newton step
                valid = np.abs(deriv) > 1e-14
                c_new = c.copy()
                c_new[valid] = c[valid] - residual[valid] / deriv[valid]
                c_guess = np.clip(c_new, 1e-12, m_vals)

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

        if time.time() > deadline - 15.0:
            break

    # Extract policy on original resource_grid
    consumption = np.zeros((horizon, Z, M))
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.interp(resource_grid, fine_grid, consumption_fine[t, z, :])

    # Enforce terminal condition
    consumption[horizon - 1, :, :] = resource_grid[None, None, :]

    # Enforce constraints: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            m_vals = resource_grid[None, :]
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_vals + 1e-10)

    return consumption
