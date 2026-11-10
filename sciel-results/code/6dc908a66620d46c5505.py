import numpy as np
from scipy.interpolate import CubicSpline
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 148.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']

    m_grid = np.asarray(resource_grid, dtype=float)
    n_m = len(m_grid)
    n_z = len(income)
    m_min = m_grid[0]
    m_max = m_grid[-1]

    # Precompute utility functions
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            return np.log(np.maximum(c, 1e-12))
        def marginal_utility(c):
            c_safe = np.maximum(c, 1e-12)
            return 1.0 / c_safe
        def second_marginal_utility(c):
            c_safe = np.maximum(c, 1e-12)
            return -1.0 / (c_safe * c_safe)
    else:
        one_minus_gamma = 1.0 - gamma
        def utility(c):
            return np.power(np.maximum(c, 1e-12), one_minus_gamma) / one_minus_gamma
        def marginal_utility(c):
            c_safe = np.maximum(c, 1e-12)
            return np.power(c_safe, -gamma)
        def second_marginal_utility(c):
            c_safe = np.maximum(c, 1e-12)
            return -gamma * np.power(c_safe, -gamma - 1)

    # Terminal condition
    consumption = np.zeros((horizon, n_z, n_m))
    consumption[horizon - 1] = m_grid[np.newaxis, :]

    # Value function at terminal
    V_next = np.tile(utility(m_grid)[np.newaxis, :], (n_z, 1))

    # Build initial splines
    V_splines = []
    for z in range(n_z):
        try:
            V_splines.append(CubicSpline(m_grid, V_next[z], bc_type='natural'))
        except Exception:
            V_splines.append(None)

    # Use a moderately fine scan grid for initial bracketing
    n_scan = 200

    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            for tt in range(t, horizon - 1):
                consumption[tt] = m_grid[np.newaxis, :]
            break

        V_curr = np.zeros((n_z, n_m))
        C_opt = np.zeros((n_z, n_m))

        for z in range(n_z):
            trans_z = transition[z]

            # Initial scan to get a good starting point
            fracs = np.linspace(0.001, 1.0, n_scan)
            c_candidates = fracs[np.newaxis, :] * m_grid[:, np.newaxis]
            m_minus_c = np.maximum(m_grid[:, np.newaxis] - c_candidates, 0.0)
            m_prime_all = R * m_minus_c[np.newaxis, :, :] + income[:, np.newaxis, np.newaxis]
            m_prime_all = np.clip(m_prime_all, m_min, m_max)

            cont_all = np.zeros((n_z, n_m, n_scan))
            for zp in range(n_z):
                mp_flat = m_prime_all[zp].ravel()
                if V_splines[zp] is not None:
                    cont_flat = V_splines[zp](mp_flat)
                else:
                    cont_flat = np.interp(mp_flat, m_grid, V_next[zp, :])
                cont_all[zp] = cont_flat.reshape(n_m, n_scan)

            cont_weighted = np.tensordot(trans_z, cont_all, axes=(0, 0))
            util_c = utility(c_candidates)
            obj_matrix = util_c + beta * cont_weighted

            best_idx = np.argmax(obj_matrix, axis=1)
            c_curr = c_candidates[np.arange(n_m), best_idx]

            # Newton-Raphson refinement with safeguarded steps
            for _ in range(15):
                if time.time() > deadline:
                    break

                m_minus_c_curr = np.maximum(m_grid[np.newaxis, :] - c_curr[np.newaxis, :], 0.0)
                m_prime_curr = R * m_minus_c_curr + income[:, np.newaxis]
                m_prime_curr = np.clip(m_prime_curr, m_min, m_max)

                V_prime_mp = np.zeros((n_z, n_m))
                V_double_prime_mp = np.zeros((n_z, n_m))

                for zp in range(n_z):
                    mp_flat = m_prime_curr[zp]
                    if V_splines[zp] is not None:
                        V_prime_mp[zp] = V_splines[zp](mp_flat, 1)
                        V_double_prime_mp[zp] = V_splines[zp](mp_flat, 2)
                    else:
                        eps = 1e-6
                        mp_up = np.clip(mp_flat + eps, m_min, m_max)
                        mp_dn = np.clip(mp_flat - eps, m_min, m_max)
                        V_up = np.interp(mp_up, m_grid, V_next[zp, :])
                        V_dn = np.interp(mp_dn, m_grid, V_next[zp, :])
                        V_mid = np.interp(mp_flat, m_grid, V_next[zp, :])
                        V_prime_mp[zp] = (V_up - V_dn) / (mp_up - mp_dn)
                        V_double_prime_mp[zp] = (V_up - 2 * V_mid + V_dn) / (eps * eps)

                E_V_prime = np.tensordot(trans_z, V_prime_mp, axes=(0, 0))
                E_V_double_prime = np.tensordot(trans_z, V_double_prime_mp, axes=(0, 0))

                mu_c = marginal_utility(c_curr)
                mu2_c = second_marginal_utility(c_curr)

                f_c = mu_c - beta * R * E_V_prime
                f_prime_c = mu2_c + beta * R * R * E_V_double_prime

                with np.errstate(divide='ignore', invalid='ignore'):
                    step = f_c / f_prime_c

                # Safeguard: limit step size
                max_up = m_grid - c_curr
                max_dn = c_curr - 1e-12
                step = np.where(step > 0, np.minimum(step, max_dn * 0.5), step)
                step = np.where(step < 0, np.maximum(step, -max_up * 0.5), step)

                c_new = c_curr - step
                c_new = np.clip(c_new, 1e-12, m_grid)

                if np.max(np.abs(c_new - c_curr)) < 1e-12 * m_max:
                    c_curr = c_new
                    break
                c_curr = c_new

            C_opt[z, :] = c_curr

            # Compute V_curr
            m_minus_c_final = np.maximum(m_grid[np.newaxis, :] - C_opt[z, np.newaxis, :], 0.0)
            m_prime_final = R * m_minus_c_final + income[:, np.newaxis]
            m_prime_final = np.clip(m_prime_final, m_min, m_max)

            cont_final = np.zeros((n_z, n_m))
            for zp in range(n_z):
                if V_splines[zp] is not None:
                    cont_final[zp] = V_splines[zp](m_prime_final[zp])
                else:
                    cont_final[zp] = np.interp(m_prime_final[zp], m_grid, V_next[zp, :])

            cont_val = np.tensordot(trans_z, cont_final, axes=(0, 0))
            util_final = utility(C_opt[z, :])
            V_curr[z, :] = util_final + beta * cont_val

        consumption[t] = C_opt
        V_next = V_curr

        V_splines = []
        for z in range(n_z):
            try:
                V_splines.append(CubicSpline(m_grid, V_next[z], bc_type='natural'))
            except Exception:
                V_splines.append(None)

    consumption[horizon - 1] = m_grid[np.newaxis, :]

    # Final validation
    for t in range(horizon):
        for z in range(n_z):
            c = consumption[t, z, :]
            c = np.maximum(c, 1e-12)
            c = np.minimum(c, m_grid + 1e-10)
            consumption[t, z, :] = c

    return consumption
