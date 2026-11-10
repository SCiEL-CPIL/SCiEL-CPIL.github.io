"""Consumption-saving policy via vectorized backward induction with FOC bisection refinement."""
import numpy as np
import time


# EVOLVE-BLOCK-START
def solve_consumption(model, resource_grid, budget):
    """
    Solve the consumption-saving problem via backward induction using a
    coarse scan + vectorized FOC-based bisection refinement.

    Parameters
    ----------
    model : dict with 'beta', 'gamma', 'R', 'income', 'transition', 'horizon'
    resource_grid : np.array of shape (101,)
    budget : int (number of consumption scan points per state)

    Returns
    -------
    consumption : np.array of shape (6, 2, 101)
    """
    start_time = time.time()
    deadline = start_time + 144.0  # safety margin before the 150s kill

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

    # ---- utility / marginal utility ------------------------------------
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            return np.log(np.maximum(c, 1e-12))
        def marg_util(c):
            return 1.0 / np.maximum(c, 1e-12)
    else:
        omg = 1.0 - gamma
        def utility(c):
            return np.power(np.maximum(c, 1e-12), omg) / omg
        def marg_util(c):
            return np.power(np.maximum(c, 1e-12), -gamma)

    # ---- terminal condition: consume everything at last period ---------
    consumption = np.zeros((horizon, n_z, n_m))
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    V_next = np.tile(utility(m_grid)[np.newaxis, :], (n_z, 1))

    n_scan = max(budget, 300)
    fracs = np.linspace(0.001, 1.0, n_scan)

    # ---- backward induction --------------------------------------------
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # safety: fill remaining periods with the consume-all heuristic
            for tt in range(t, horizon - 1):
                consumption[tt] = m_grid[np.newaxis, :]
            break

        # V'_next via central differences (vectorized over z)
        V_prime_next = np.empty((n_z, n_m))
        V_prime_next[:, 0] = (V_next[:, 1] - V_next[:, 0]) / (m_grid[1] - m_grid[0])
        V_prime_next[:, -1] = (V_next[:, -1] - V_next[:, -2]) / (m_grid[-1] - m_grid[-2])
        V_prime_next[:, 1:-1] = (V_next[:, 2:] - V_next[:, :-2]) / (m_grid[2:] - m_grid[:-2])

        V_curr = np.zeros((n_z, n_m))
        C_opt = np.zeros((n_z, n_m))

        for z in range(n_z):
            trans_z = transition[z]

            # ---- coarse scan to locate the optimum region --------------
            c_cand = fracs[np.newaxis, :] * m_grid[:, np.newaxis]      # (n_m, n_scan)
            m_minus_c = np.maximum(m_grid[:, np.newaxis] - c_cand, 0.0)
            mp_all = np.clip(R * m_minus_c[np.newaxis, :, :] + income[:, np.newaxis, np.newaxis],
                             m_min, m_max)                             # (n_z, n_m, n_scan)
            cont_all = np.empty((n_z, n_m, n_scan))
            for zp in range(n_z):
                cont_all[zp] = np.interp(mp_all[zp].ravel(), m_grid, V_next[zp, :]).reshape(n_m, n_scan)
            cont_w = np.tensordot(trans_z, cont_all, axes=(0, 0))      # (n_m, n_scan)
            obj = utility(c_cand) + beta * cont_w
            best_idx = np.argmax(obj, axis=1)
            best_c = c_cand[np.arange(n_m), best_idx]

            # ---- vectorized FOC bisection refinement --------------------
            # For a given c vector (n_m,), the derivative of the objective is
            #   d(c) = u'(c) - beta * R * sum_zp trans_z[zp] * V'_next[zp](mp(zp))
            # where mp(zp) = clip(R*(m - c) + income[zp], m_min, m_max).
            # Because the objective is concave in c, d is decreasing in c.
            # We bisect for the root in [c_lo, c_hi] per m point.
            c_lo = np.maximum(1e-12, best_c * 0.8)
            c_hi = np.minimum(m_grid, best_c * 1.2)
            # guard degenerate intervals
            bad = c_lo >= c_hi
            c_lo[bad] = np.maximum(1e-12, m_grid[bad] * 0.001)
            c_hi[bad] = m_grid[bad]

            def deriv(c_vec):
                c_safe = np.maximum(c_vec, 1e-12)
                m_minus = np.maximum(m_grid[np.newaxis, :] - c_safe[np.newaxis, :], 0.0)
                mp = np.clip(R * m_minus + income[:, np.newaxis], m_min, m_max)  # (n_z, n_m)
                Vp = np.empty((n_z, n_m))
                for zp in range(n_z):
                    Vp[zp] = np.interp(mp[zp], m_grid, V_prime_next[zp, :])
                return marg_util(c_safe) - beta * R * (trans_z @ Vp)  # (n_m,)

            d_lo = deriv(c_lo)
            d_hi = deriv(c_hi)
            # if derivative already <=0 at lo -> optimum at lo; >=0 at hi -> at hi
            at_lo = d_lo <= 0.0
            at_hi = d_hi >= 0.0
            interior = ~(at_lo | at_hi)

            lo = c_lo.copy()
            hi = c_hi.copy()
            if np.any(interior):
                for _ in range(50):
                    mid = 0.5 * (lo[interior] + hi[interior])
                    dm = deriv(mid)
                    up = dm[interior] > 0.0
                    # update only interior indices
                    idx = np.where(interior)[0]
                    lo[idx[up]] = mid[up]
                    hi[idx[~up]] = mid[~up]
                    if np.max(np.abs(hi[interior] - lo[interior])) < 1e-14 * m_max:
                        break

            c_opt = np.where(at_lo, c_lo, np.where(at_hi, c_hi, 0.5 * (lo + hi)))
            c_opt = np.clip(c_opt, 1e-12, m_grid)
            C_opt[z, :] = c_opt

            # ---- value at the refined optimum (vectorized) --------------
            c_safe = np.maximum(c_opt, 1e-12)
            m_minus = np.maximum(m_grid[np.newaxis, :] - c_safe[np.newaxis, :], 0.0)
            mp = np.clip(R * m_minus + income[:, np.newaxis], m_min, m_max)
            cont = np.empty((n_z, n_m))
            for zp in range(n_z):
                cont[zp] = np.interp(mp[zp], m_grid, V_next[zp, :])
            V_curr[z, :] = utility(c_safe) + beta * (trans_z @ cont)

        consumption[t] = C_opt
        V_next = V_curr

    # ---- enforce terminal condition & validity --------------------------
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    consumption = np.maximum(consumption, 1e-12)
    consumption = np.minimum(consumption, m_grid[np.newaxis, np.newaxis, :] + 1e-10)

    return consumption
# EVOLVE-BLOCK-END
