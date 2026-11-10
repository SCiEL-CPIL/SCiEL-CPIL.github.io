import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    T = horizon
    Z = len(income)
    N_out = len(resource_grid)

    # Dense internal grid in log-space for accurate value function
    n_dense = 2000
    m_min = resource_grid[0]
    m_max = resource_grid[-1]
    log_grid = np.linspace(np.log(m_min), np.log(m_max), n_dense)
    dense_grid = np.exp(log_grid)

    def util_prime(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return 1.0 / c
        else:
            return c ** (-gamma)

    def util_second(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return -1.0 / (c * c)
        else:
            return -gamma * c ** (-gamma - 1)

    # c_policy[t, z, i] for output grid
    c_policy = np.zeros((T, Z, N_out))

    # Last period: consume everything
    c_policy[T - 1, :, :] = resource_grid[np.newaxis, :, :]

    # For each period from T-2 down to 0
    # Store value function V[t, z, :] on dense grid
    # and consumption policy on dense grid for interpolation

    # We'll store the value function and its derivative on the dense grid
    # V[t, z, :] shape (Z, n_dense)
    # dV[t, z, :] shape (Z, n_dense)

    # At t = T-1: V = u(m), dV = u'(m)
    V_next = np.zeros((Z, n_dense))
    dV_next = np.zeros((Z, n_dense))
    for z in range(Z):
        V_next[z, :] = np.log(np.maximum(dense_grid, 1e-12)) if np.isclose(gamma, 1.0) else \
            (np.maximum(dense_grid, 1e-12) ** (1 - gamma)) / (1 - gamma)
        dV_next[z, :] = util_prime(dense_grid)

    # Store consumption on dense grid for each period for final interpolation
    c_dense = np.zeros((T, Z, n_dense))
    c_dense[T - 1, :, :] = dense_grid[np.newaxis, :, :]

    for t in range(T - 2, -1, -1):
        if time.time() > deadline:
            break

        V_curr = np.zeros((Z, n_dense))
        dV_curr = np.zeros((Z, n_dense))
        c_curr_dense = np.zeros((Z, n_dense))

        for z in range(Z):
            # For each income state z, solve Euler equation for each m in dense_grid
            # Euler: u'(c) = beta * sum_{z'} P(z'|z) * [ R * dV_next(z', m') ]
            # where m' = R*(m - c) + income[z']

            # We need to find c such that:
            # u'(c) = beta * sum_{z'} P(z'|z) * R * dV_next(z', R*(m-c) + income[z'])

            # Newton-Raphson: f(c) = u'(c) - beta * sum_{z'} P(z'|z) * R * dV_next(z', R*(m-c)+income[z'])
            # f'(c) = u''(c) + beta * sum_{z'} P(z'|z) * R^2 * d2V_next(z', R*(m-c)+income[z'])

            # We need d2V_next. For the last period, d2V = u''(m).
            # For intermediate periods, we can compute numerically or analytically.

            # Actually, let's use a different approach: for each m, do a 1D Newton solve
            # but we need dV_next as a function of m' which requires interpolation.

            # Build PCHIP interpolators for dV_next for each z'
            pchip_dV = []
            for zp in range(Z):
                pchip_dV.append(PchipInterpolator(dense_grid, dV_next[zp, :]))

            # For each m in dense_grid, solve for c
            c_solutions = np.zeros(n_dense)

            for i in range(n_dense):
                m = dense_grid[i]
                if m <= 0:
                    c_solutions[i] = 1e-12
                    continue

                # Initial guess: proportional consumption
                c = min(0.8 * m, m)  # start with 80% of wealth
                c = max(c, 1e-12)

                for it in range(50):
                    # Compute m' for each z'
                    m_primes = R * (m - c) + income  # shape (Z,)
                    m_primes = np.maximum(m_primes, 1e-12)

                    # dV_next at m' for each z'
                    dV_vals = np.zeros(Z)
                    for zp in range(Z):
                        # Interpolate dV_next[zp] at m'
                        if m_primes[zp] <= dense_grid[0]:
                            dV_vals[zp] = dV_next[zp, 0]
                        elif m_primes[zp] >= dense_grid[-1]:
                            dV_vals[zp] = dV_next[zp, -1]
                        else:
                            dV_vals[zp] = pchip_dV[zp](m_primes[zp])

                    # RHS of Euler
                    rhs = beta * np.dot(transition[z, :], R * dV_vals)

                    # f(c) = u'(c) - rhs
                    uc = util_prime(c)
                    f = uc - rhs

                    # f'(c) = u''(c) + beta * sum P(z'|z) * R^2 * d2V_next(z', m')
                    # We need d2V_next. Let's approximate it numerically from dV_next.
                    # Or use the analytical form: for CRRA, if V is CRRA-like, d2V = -gamma * dV / m
                    # Actually, let's just use numerical differentiation for d2V.

                    # Numerical d2V: use central difference on the dense grid
                    # d2V_next[zp, i] ≈ (dV_next[zp, i+1] - 2*dV_next[zp, i] + dV_next[zp, i-1]) / dm^2
                    # But m' may not be on the grid. Let's interpolate d2V.

                    # Precompute d2V_next on dense grid
                    # Actually, let's just use a simple approach: approximate d2V as -gamma * dV / m' for CRRA
                    # This is exact for the last period and approximately right for others.

                    d2V_vals = np.zeros(Z)
                    for zp in range(Z):
                        mp = m_primes[zp]
                        if mp <= 1e-12:
                            d2V_vals[zp] = 0.0
                        else:
                            # Approximate: for CRRA, d2V ≈ -gamma * dV / m
                            d2V_vals[zp] = -gamma * dV_vals[zp] / mp

                    fp = util_second(c) + beta * np.dot(transition[z, :], R * R * d2V_vals)

                    if abs(fp) < 1e-30:
                        break

                    c_new = c - f / fp
                    c_new = max(c_new, 1e-12)
                    c_new = min(c_new, m)  # c cannot exceed m

                    if abs(c_new - c) < 1e-12 * max(1.0, c):
                        c = c_new
                        break
                    c = c_new

                c_solutions[i] = c

            c_curr_dense[z, :] = c_solutions

            # Compute V_curr and dV_curr using the solved policy
            # V(m) = u(c) + beta * sum_{z'} P(z'|z) * V_next(z', R*(m-c)+income[z'])
            # dV(m) = u'(c) * dc/dm ... but we can compute directly:
            # Actually, from the envelope theorem: dV/dm = u'(c) * (dc/dm) + ... 
            # Simpler: V(m) = u(c(m)) + beta * sum P * V_next(z', m')
            # dV/dm = u'(c) * c'(m) + beta * sum P * dV_next(z', m') * R
            # But from Euler: u'(c) = beta * sum P * R * dV_next(z', m')
            # So dV/dm = u'(c) * c'(m) + u'(c) = u'(c) * (1 + c'(m))... 
            # Hmm, this is getting complicated. Let me just compute V directly and use PCHIP for dV.

            V_vals = np.zeros(n_dense)
            for i in range(n_dense):
                m = dense_grid[i]
                c = c_solutions[i]
                if np.isclose(gamma, 1.0):
                    u_c = np.log(max(c, 1e-12))
                else:
                    u_c = (max(c, 1e-12) ** (1 - gamma)) / (1 - gamma)

                m_primes = R * (m - c) + income
                m_primes = np.maximum(m_primes, 1e-12)

                V_next_vals = np.zeros(Z)
                for zp in range(Z):
                    if m_primes[zp] <= dense_grid[0]:
                        V_next_vals[zp] = V_next[zp, 0]
                    elif m_primes[zp] >= dense_grid[-1]:
                        V_next_vals[zp] = V_next[zp, -1]
                    else:
                        # Linear interpolation for V (or use PCHIP)
                        idx = np.searchsorted(dense_grid, m_primes[zp])
                        idx = min(idx, n_dense - 1)
                        idx0 = max(idx - 1, 0)
                        if idx0 == idx:
                            V_next_vals[zp] = V_next[zp, idx]
                        else:
                            w = (m_primes[zp] - dense_grid[idx0]) / (dense_grid[idx + 1] - dense_grid[idx0])
                            V_next_vals[zp] = (1 - w) * V_next[zp, idx0] + w * V_next[zp, idx + 1]

                V_vals[i] = u_c + beta * np.dot(transition[z, :], V_next_vals)

            V_curr[z, :] = V_vals

            # Compute dV_curr using PCHIP on V_curr
            pchip_V = PchipInterpolator(dense_grid, V_vals)
            dV_curr[z, :] = pchip_V(dense_grid, 1)

        # Set up for next iteration
        V_next = V_curr
        dV_next = dV_curr
        c_dense[t, :, :] = c_curr_dense

        # Extract policy on output grid for this period
        for z in range(Z):
            # Interpolate c_curr_dense[z, :] from dense_grid to resource_grid
            if resource_grid[0] >= dense_grid[0] and resource_grid[-1] <= dense_grid[-1]:
                c_policy[t, z, :] = np.interp(resource_grid, dense_grid, c_curr_dense[z, :])
            else:
                c_policy[t, z, :] = np.interp(resource_grid, dense_grid, c_curr_dense[z, :],
                                              left=c_curr_dense[z, 0], right=c_curr_dense[z, -1])

    # Final checks and clipping
    c_policy = np.clip(c_policy, 1e-12, None)
    for t in range(T):
        for z in range(Z):
            c_policy[t, z, :] = np.minimum(c_policy[t, z, :], resource_grid + 1e-10)
            c_policy[t, z, :] = np.maximum(c_policy[t, z, :], 1e-12)

    # Ensure last period consumes everything
    c_policy[T - 1, :, :] = resource_grid[np.newaxis, :, :]

    # Ensure finite values
    c_policy = np.where(np.isfinite(c_policy), c_policy, 1e-12)

    return c_policy
