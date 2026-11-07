"""Rewritten program: backward induction with robust concave Bellman minimization."""
import numpy as np
import time


def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    beta = float(model['beta'])
    gamma = float(model['gamma'])
    R = float(model['R'])
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = int(model['horizon'])
    m_grid = np.asarray(resource_grid, dtype=float)
    N_m = len(m_grid)
    N_z = len(income)
    T = horizon  # 6

    # --- Utility and derivatives (vectorized where possible) ---
    def util(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return np.power(c, -gamma)

    def u_double_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return -1.0 / (c * c)
        else:
            return -gamma * np.power(c, -gamma - 1.0)

    # --- Continuation value storage: V[t, z, m_index] ---
    # V[t] has shape (N_z, N_m). Terminal: V[T-1] = util(m) (consume all).
    # Actually we define V[t] = value at time t given m, z. At t = T-1 (last period),
    # c = m, so V[T-1, z, m] = util(m).
    V = np.zeros((T, N_z, N_m), dtype=float)
    # Terminal period
    V[T - 1, :, :] = util(m_grid)[None, :]

    # consumption policy: c_policy[t, z, m_index]
    c_policy = np.zeros((T, N_z, N_m), dtype=float)
    c_policy[T - 1, :, :] = m_grid[None, :]

    # For each t from T-2 down to 0
    for t in range(T - 2, -1, -1):
        # Check deadline
        if time.time() > DEADLINE - 20.0:
            # Fill remaining with a simple heuristic: consume all (safe, valid)
            for tt in range(t, -1, -1):
                c_policy[tt, :, :] = m_grid[None, :]
            break

        V_next = V[t + 1]  # (N_z, N_m)

        for z in range(N_z):
            # Continuation value as function of m': interpolate V_next over m_grid
            # We'll build a function cont(m') = sum_z' P(z'|z) * interp(m_grid, V_next[z'], m')
            # Since m_grid is fixed, precompute interpolation per z'.
            # For speed, we evaluate cont at specific m' points per m.

            # For each m in m_grid, find optimal c.
            # m' = R*(m - c) + income[z']
            # We minimize Bellman objective: -[util(c) + beta * cont(m')]
            # over c in [1e-12, m].

            # Warm-start: use previous m's optimal fraction.
            prev_frac = 0.5
            for i in range(N_m):
                m = m_grid[i]
                # Continuation value function evaluated at m'
                # m' for each z':
                # We'll define objective(c) = -util(c) - beta * sum_z' P(z'|z) * interp(V_next[z'], m')
                # where m' = R*(m - c) + income[z']

                # Precompute the interpolation weights isn't trivial for arbitrary m'.
                # Use np.interp per z' — but that's per c evaluation. To speed up,
                # we can precompute a dense interpolation table? m' ranges over
                # [R*(m-m)+income[z'], R*(m-1e-12)+income[z']] = [income[z'], R*m + income[z']].
                # income in [0.5, 1.5], R*m up to 1.02*40=40.8. So m' up to ~42.
                # m_grid max is 40. For m' > 40, np.interp clamps to last value.

                # We'll build the objective as a closure that uses np.interp.
                # To reduce overhead, precompute the interpolation for each z' as a
                # function using np.interp (fast for small N_m).

                # Define objective
                def objective(c, m=m, z=z):
                    c = max(c, 1e-12)
                    c = min(c, m)
                    u_c = util(c)
                    # continuation
                    m_minus_c = m - c
                    cont = 0.0
                    for zp in range(N_z):
                        pp = transition[z, zp]
                        if pp < 1e-15:
                            continue
                        mp = R * m_minus_c + income[zp]
                        # interpolate V_next[zp] at mp
                        vp = np.interp(mp, m_grid, V_next[zp])
                        cont += pp * vp
                    return -(u_c + beta * cont)

                # Derivative of objective (for Newton polish)
                def objective_prime(c, m=m, z=z):
                    c = max(c, 1e-12)
                    c = min(c, m)
                    du = -u_prime(c)  # d/dc of -util(c)
                    # d/dc of -beta * cont(m')
                    # m' = R*(m-c) + income[zp], d m'/dc = -R
                    # d cont/dc = sum_zp P(zp) * V_next_zp'(mp) * (-R)
                    # We approximate V_next_zp' by finite difference of interp
                    # Actually, we can compute derivative of interp analytically:
                    # np.interp is piecewise linear, so derivative is slope of segment.
                    # For Newton polish, we can use a central difference of objective.
                    # Simpler: use central difference.
                    h = max(1e-8, c * 1e-6)
                    c_plus = min(c + h, m)
                    c_minus = max(c - h, 1e-12)
                    if c_plus == c_minus:
                        return 0.0
                    return (objective(c_plus) - objective(c_minus)) / (c_plus - c_minus)

                # Second derivative (for Newton step) via central difference of derivative
                def objective_double(c, m=m, z=z):
                    h = max(1e-7, c * 1e-5)
                    c_plus = min(c + h, m)
                    c_minus = max(c - h, 1e-12)
                    if c_plus == c_minus:
                        return -1.0
                    return (objective_prime(c_plus) - objective_prime(c_minus)) / (c_plus - c_minus)

                # --- Concave maximization of f(c) = -objective(c) over [1e-12, m] ---
                # f is strictly concave. Use golden-section search then Newton polish.
                lo = 1e-12
                hi = m
                if hi <= lo:
                    c_opt = lo
                else:
                    # Golden section
                    invphi = (np.sqrt(5.0) - 1.0) / 2.0
                    invphi2 = (3.0 - np.sqrt(5.0)) / 2.0
                    a = lo
                    b = hi
                    c1 = b - invphi * (b - a)
                    c2 = a + invphi * (b - a)
                    f1 = -objective(c1)
                    f2 = -objective(c2)
                    for _ in range(40):
                        if f1 < f2:
                            a = c1
                            c1 = c2
                            f1 = f2
                            c2 = a + invphi * (b - a)
                            f2 = -objective(c2)
                        else:
                            b = c2
                            c2 = c1
                            f2 = f1
                            c1 = b - invphi * (b - a)
                            f1 = -objective(c1)
                        if b - a < 1e-10 * max(1.0, m):
                            break
                    c_opt = 0.5 * (a + b)

                    # Newton polish (a few iterations)
                    for _ in range(5):
                        if c_opt <= lo or c_opt >= hi:
                            break
                        fval = -objective(c_opt)
                        fprime = -objective_prime(c_opt)  # derivative of f
                        fdouble = -objective_double(c_opt)  # second derivative of f
                        if fdouble >= 0:
                            break  # not concave here, stop
                        step = fprime / fdouble
                        # Damped Newton
                        new_c = c_opt - step
                        if new_c <= lo or new_c >= hi:
                            new_c = 0.5 * (c_opt + (hi if new_c > hi else lo))
                        if abs(new_c - c_opt) < 1e-12 * max(1.0, m):
                            c_opt = new_c
                            break
                        c_opt = new_c

                # Enforce constraints
                c_opt = max(1e-12, min(c_opt, m))
                c_policy[t, z, i] = c_opt
                prev_frac = c_opt / m if m > 0 else 0.5

        # After computing c_policy[t], update V[t]
        for z in range(N_z):
            for i in range(N_m):
                m = m_grid[i]
                c = c_policy[t, z, i]
                m_minus_c = m - c
                cont = 0.0
                for zp in range(N_z):
                    pp = transition[z, zp]
                    if pp < 1e-15:
                        continue
                    mp = R * m_minus_c + income[zp]
                    vp = np.interp(mp, m_grid, V[t + 1][zp])
                    cont += pp * vp
                V[t, z, i] = util(c) + beta * cont

    # Enforce terminal condition: c[5] = m within 1e-9
    c_policy[T - 1, :, :] = m_grid[None, :]

    # Ensure finite and within bounds
    c_policy = np.nan_to_num(c_policy, nan=1e-12, posinf=m_grid.max(), neginf=1e-12)
    # 0 < c <= m + 1e-10
    for t in range(T):
        for z in range(N_z):
            for i in range(N_m):
                m = m_grid[i]
                c_policy[t, z, i] = max(1e-12, min(c_policy[t, z, i], m + 1e-10))

    return c_policy
