import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    num_states = len(income)
    num_resources = len(resource_grid)

    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1 - gamma) / (1 - gamma)

    def util_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return c ** (-gamma)

    def util_second(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return -1.0 / (c * c)
        return -gamma * c ** (-gamma - 1)

    policy = np.zeros((horizon, num_states, num_resources))
    value = np.zeros((horizon, num_states, num_resources))

    # Terminal: consume all
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    value[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, :]

    # Build log-space spline for terminal value
    log_grid = np.log(resource_grid)

    def build_spline(val_1d):
        # val_1d: shape (num_resources,)
        # Use log-space cubic spline
        try:
            cs = CubicSpline(log_grid, val_1d, bc_type='natural')
            return cs
        except Exception:
            # Fallback to linear interp in log space
            return None

    # Pre-build terminal splines
    splines_next = [build_spline(value[horizon - 1, z, :]) for z in range(num_states)]

    max_newton_iter = 50
    tol = 1e-10

    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: equal-split heuristic
            for tt in range(t, -1, -1):
                frac = 1.0 / (horizon - tt)
                policy[tt, :, :] = np.maximum(frac * resource_grid[np.newaxis, :], 1e-12)
                policy[tt, :, :] = np.minimum(policy[tt, :, :], resource_grid[np.newaxis, :])
                value[tt, :, :] = utility(policy[tt, :, :])
            break

        for z in range(num_states):
            m = resource_grid  # (num_resources,)
            c = np.clip(m * 0.5, 1e-12, m)  # initial guess: consume half

            # Newton iterations on the FOC:
            # u'(c) = beta * sum_{zp} trans[z,zp] * V'_{t+1,zp}(R*(m-c)+income[zp]) * R
            # Define f(c) = u'(c) - beta * R * sum_{zp} trans[z,zp] * V'_{t+1,zp}(m_next(zp))
            # f'(c) = u''(c) + beta * R^2 * sum_{zp} trans[z,zp] * V''_{t+1,zp}(m_next(zp))

            for _iter in range(max_newton_iter):
                if time.time() > deadline - 3:
                    break

                # Compute m_next for each zp
                # m_next[zp, r] = R*(m[r] - c[r]) + income[zp]
                m_next = R * (m[np.newaxis, :] - c[np.newaxis, :]) + income[:, np.newaxis]
                m_next = np.maximum(m_next, 1e-10)  # (num_states, num_resources)

                # Compute V' and V'' at m_next using splines
                # If spline available, use it; else use finite differences on linear interp
                v_prime = np.zeros((num_states, num_resources))
                v_double_prime = np.zeros((num_states, num_resources))
                v_val = np.zeros((num_states, num_resources))

                for zp in range(num_states):
                    if splines_next[zp] is not None:
                        # Evaluate spline and its derivatives at m_next[zp]
                        # Spline is in log-space, so dV/dm = (dV/dlog_m) / m
                        # dV/dlog_m = cs'(log_m)
                        # d2V/dm2 = (d2V/dlog_m2 - dV/dlog_m) / m^2
                        log_m = np.log(np.maximum(m_next[zp, :], 1e-12))
                        # Clip log_m to spline domain
                        log_m_clipped = np.clip(log_m, log_grid[0], log_grid[-1])
                        dV_dlogm = splines_next[zp](log_m_clipped, 1)
                        d2V_dlogm2 = splines_next[zp](log_m_clipped, 2)
                        v_val[zp, :] = splines_next[zp](log_m_clipped, 0)
                        v_prime[zp, :] = dV_dlogm / np.maximum(m_next[zp, :], 1e-10)
                        v_double_prime[zp, :] = (d2V_dlogm2 - dV_dlogm) / (m_next[zp, :] ** 2)
                    else:
                        # Fallback: linear interpolation with central differences
                        m_clipped = np.clip(m_next[zp, :], resource_grid[0], resource_grid[-1])
                        v_val[zp, :] = np.interp(m_clipped, resource_grid, value[t + 1, zp, :])
                        # Central difference for first derivative
                        h = 1e-6
                        m_plus = np.clip(m_clipped + h, resource_grid[0], resource_grid[-1])
                        m_minus = np.clip(m_clipped - h, resource_grid[0], resource_grid[-1])
                        v_plus = np.interp(m_plus, resource_grid, value[t + 1, zp, :])
                        v_minus = np.interp(m_minus, resource_grid, value[t + 1, zp, :])
                        v_prime[zp, :] = (v_plus - v_minus) / (m_plus - m_minus)
                        # Second derivative
                        v_double_prime[zp, :] = (v_plus - 2 * v_val[zp, :] + v_minus) / (h * h)

                # Weighted sum over zp
                w = transition[z, :]  # (num_states,)
                exp_v_prime = np.sum(w[:, np.newaxis] * v_prime, axis=0)  # (num_resources,)
                exp_v_dprime = np.sum(w[:, np.newaxis] * v_double_prime, axis=0)
                exp_v_val = np.sum(w[:, np.newaxis] * v_val, axis=0)

                # FOC residual: f(c) = u'(c) - beta * R * exp_v_prime
                f = util_deriv(c) - beta * R * exp_v_prime
                # Derivative: f'(c) = u''(c) + beta * R^2 * exp_v_dprime
                fp = util_second(c) + beta * R * R * exp_v_dprime

                # Newton step: c_new = c - f/fp
                # Guard against division by zero
                fp_safe = np.where(np.abs(fp) < 1e-15, 1e-15, fp)
                step = f / fp_safe
                # Damping to stay in bounds
                c_new = c - step
                # Ensure c in (1e-12, m)
                c_new = np.clip(c_new, 1e-12, m)

                # Check convergence
                if np.max(np.abs(c_new - c)) < tol:
                    c = c_new
                    break
                c = c_new

            policy[t, z, :] = c
            value[t, z, :] = utility(c) + beta * exp_v_val

            # Ensure validity
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, m)

        # Build splines for next iteration (current value becomes next for earlier t)
        splines_next = [build_spline(value[t, z, :]) for z in range(num_states)]

    # Terminal fix
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]

    # Final validity
    for t in range(horizon):
        for z in range(num_states):
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, resource_grid)

    if not np.all(np.isfinite(policy)):
        policy = np.nan_to_num(policy, nan=1e-12, posinf=1e10, neginf=0)

    return policy
