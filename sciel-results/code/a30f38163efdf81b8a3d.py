import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M_out = len(resource_grid)
    Z = len(income)

    # Use a fine internal grid for high accuracy
    M_int = 401
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
    # We store the policy as c directly.

    # Initialize the "next" policy function (for t=5)
    # c_next[z, m_idx] on the internal grid
    c_next = np.zeros((Z, M_int))
    for z in range(Z):
        c_next[z] = m_grid

    # Set terminal period consumption on output grid
    for z in range(Z):
        consumption[horizon - 1, z] = resource_grid

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
            # For gamma=1 (log utility), optimal s is often around 0.2-0.3.
            # For gamma>1, saving is lower.
            # Let's start with s=0.2 for all m.
            s = np.full(M_int, 0.2)

            # Newton's method iteration
            # We need F(s) and F'(s)
            # F(s) = -gamma * log(c) - log( beta * R * sum P * u'(c_next) )

            # Pre-compute transition weights for this z
            # P_z = transition[z, :]

            for _ in range(15):  # Max 15 iterations, usually converges in 3-5
                if time.time() > deadline:
                    break

                c = m_grid * (1.0 - s)
                # Ensure c > 0
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
                # For each z', we have c_next[z'] (M_int,) and we evaluate at m_prime[:, z'] (M_int,)

                u_prime_next_sum = np.zeros(M_int)
                for zp in range(Z):
                    # m_prime[:, zp] is the m' for state zp
                    # Interpolate c_next[zp] at m_prime[:, zp]
                    c_next_val = np.interp(m_prime[:, zp], m_grid, c_next[zp])
                    # Compute u'(c_next_val)
                    u_prime_val = u_prime(c_next_val, gamma)
                    # Weight by transition probability
                    u_prime_next_sum += transition[z, zp] * u_prime_val

                # rhs_val = beta * R * u_prime_next_sum
                rhs_val = beta * R * u_prime_next_sum
                # Ensure positive for log
                rhs_val = np.maximum(rhs_val, 1e-12)
                rhs = np.log(rhs_val)

                # F(s) = lhs - rhs
                F = lhs - rhs

                # Derivative F'(s)
                # d/ds log(u'(c)) where c = m(1-s)
                # u'(c) = c^{-gamma}
                # log u'(c) = -gamma log(c)
                # d/ds (-gamma log(m(1-s))) = -gamma * (1/(m(1-s))) * (-m) = gamma / (1-s)
                # So d/ds lhs = gamma / (1 - s) (for gamma != 1)
                # For gamma = 1, log u'(c) = -log(c), derivative is 1/(1-s).
                # So d/ds lhs = gamma / (1 - s) works for all gamma?
                # Let's check:
                # u'(c) = c^{1-gamma-1} = c^{-gamma}
                # log u' = -gamma log c
                # d/ds = -gamma * (1/c) * dc/ds
                # c = m(1-s), dc/ds = -m
                # d/ds = -gamma * (1/(m(1-s))) * (-m) = gamma / (1-s)
                # Yes, this holds for any gamma.

                d_lhs = gamma / (1.0 - s)

                # d/ds rhs = d/ds log( beta R sum P u'(c_next) )
                # = (1 / (beta R sum P u')) * beta R sum P d/ds u'(c_next)
                # = (1 / E[u']) * E[ d/ds u'(c_next) ]

                # u'(c_next) = c_next^{-gamma}
                # d/ds u'(c_next) = -gamma * c_next^{-gamma-1} * dc_next/ds
                # c_next is a function of m' = R m s + y
                # dc_next/ds = dc_next/dm' * dm'/ds
                # dm'/ds = R m
                # We need the derivative of the interpolated c_next.
                # Approximation: Numerical derivative or analytic if we had a parametric form.
                # Since c_next is piecewise linear (from np.interp), its derivative is piecewise constant (slope of segment).
                # However, calculating the slope of the interp segment for each point is complex.
                # Alternative: Use a numerical derivative for F(s) to avoid complex analytic derivs of the interp.
                # Newton with numerical derivative is "Secant" or just use a small step.
                # Or, simpler: use a robust root finding like Bisection + Newton hybrid?
                # Given the smoothness, a simple numerical derivative step size h=1e-4 might work.

                # Let's use a numerical derivative for the whole F(s) to be safe and simple.
                h = 1e-5
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
                # Guard against division by zero
                dF_safe = np.where(np.abs(dF) < 1e-10, 1e-10, dF)
                s_new = s - F / dF_safe

                # Clip s to valid range [0, 0.999999]
                s_new = np.clip(s_new, 0.0, 0.999999)

                # Check convergence
                if np.max(np.abs(s_new - s)) < 1e-8:
                    s = s_new
                    break
                s = s_new

            # Compute c_curr from s
            c_curr[z] = m_grid * (1.0 - s)
            c_curr[z] = np.maximum(c_curr[z], 1e-12)
            c_curr[z] = np.minimum(c_curr[z], m_grid + 1e-10)

        # Store the policy for this period on the output grid
        for z in range(Z):
            consumption[t, z] = np.interp(resource_grid, m_grid, c_curr[z])

        # Update c_next for the next iteration (backwards)
        c_next = c_curr.copy()

    # Final validity checks
    for t in range(horizon):
        for z in range(Z):
            c = consumption[t, z]
            m = resource_grid
            # 0 < c
            c = np.maximum(c, 1e-12)
            # c <= m + 1e-10
            c = np.minimum(c, m + 1e-10)
            consumption[t, z] = c

    # Enforce terminal condition exactly
    for z in range(Z):
        consumption[horizon - 1, z] = resource_grid

    return consumption
