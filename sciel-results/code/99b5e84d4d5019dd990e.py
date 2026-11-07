"""Rewritten from scratch: FOC bisection value iteration for consumption-saving policy."""
import numpy as np


def _u(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return np.log(c)
    return np.power(c, 1.0 - gamma) / (1.0 - gamma)


def _up(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return 1.0 / c
    return np.power(c, -gamma)


def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    trans = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    n_states = len(income)
    mgrid = np.asarray(resource_grid, dtype=float)
    n = len(mgrid)

    # Terminal: consume everything -> V_h(m, z) = u(m) for all z
    V = np.tile(_u(mgrid, gamma)[None, :], (n_states, 1))   # (n_states, n)
    c_policy = np.zeros((horizon, n_states, n))
    c_policy[horizon - 1] = np.tile(mgrid[None, :], (n_states, 1))

    # Log-spaced search grid for the policy function c*(m)
    m_search = np.exp(np.linspace(np.log(1e-12), np.log(40.0), 2048))
    n_log = len(m_search)

    for t in range(horizon - 2, -1, -1):
        V_next = V
        for z in range(n_states):
            c_on_search = np.empty(n_log)
            for i in range(n_log):
                m = m_search[i]

                def fof(c):
                    c = min(max(c, 1e-12), m + 1e-10)
                    lhs = _up(c, gamma)
                    rhs = 0.0
                    for zeta in range(n_states):
                        p = trans[z, zeta]
                        if p <= 0:
                            continue
                        mp = R * (m - c) + income[zeta]
                        cz = max(mp, 1e-12)  # u increasing -> consume all of m'
                        rhs += p * R * _up(cz, gamma)
                    return lhs - beta * rhs

                lo, hi = 1e-12, m + 1e-10
                if fof(lo) < 0:
                    c_on_search[i] = lo
                    continue
                if fof(hi) > 0:
                    c_on_search[i] = hi
                    continue
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    if fof(mid) > 0:
                        lo = mid
                    else:
                        hi = mid
                c_on_search[i] = 0.5 * (lo + hi)

            c_interp = np.interp(mgrid, m_search, c_on_search)
            c_interp = np.clip(c_interp, 1e-12, mgrid + 1e-10)
            c_policy[t, z, :] = c_interp

            Vz = np.empty(n)
            for i in range(n):
                m = mgrid[i]
                c = c_interp[i]
                val = _u(c, gamma)
                for zeta in range(n_states):
                    p = trans[z, zeta]
                    if p <= 0:
                        continue
                    mp = R * (m - c) + income[zeta]
                    vnext = _u(1e-12, gamma) if mp <= 1e-12 else np.interp(mp, mgrid, V_next[zeta])
                    val += beta * p * vnext
                Vz[i] = val
            V[z, :] = Vz

    c_policy[horizon - 1] = np.tile(mgrid[None, :], (n_states, 1))
    c_policy = np.minimum(c_policy, mgrid[None, None, :] + 1e-10)
    c_policy = np.maximum(c_policy, 1e-12)
    return c_policy
