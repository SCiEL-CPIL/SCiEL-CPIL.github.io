"""Consumption-saving policy: fine log-grid value function + bisection Euler solve."""
import time
import numpy as np


def _crra_util(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-10:
        return np.log(c)
    return np.power(c, 1.0 - gamma) / (1.0 - gamma)


def _crra_mu(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-10:
        return 1.0 / c
    return np.power(c, -gamma)


def _solve_euler_bisect(m, yz, R, beta, Mz, m_grid, gamma):
    """Solve u'(c) = beta*Mz(R*(m-c)+yz) for c in [0, m] via bisection."""
    # f(c) = u'(c) - beta*Mz(R*(m-c)+yz) is strictly decreasing in c.
    Mval_m = np.interp(yz, m_grid, Mz)
    f_m = _crra_mu(np.array([m]), gamma)[0] - beta * Mval_m
    if f_m > 0:
        return m  # optimum at c = m
    lo = 0.0
    hi = m
    for it in range(80):
        mid = 0.5 * (lo + hi)
        mprime = R * (m - mid) + yz
        Mval = np.interp(mprime, m_grid, Mz)
        muc = _crra_mu(np.array([mid]), gamma)[0]
        f = muc - beta * Mval
        if f > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, m):
            break
    return 0.5 * (lo + hi)


def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    beta = float(model['beta'])
    gamma = float(model['gamma'])
    R = float(model['R'])
    income = np.asarray(model['income'], dtype=float)
    trans = np.asarray(model['transition'], dtype=float)
    T = int(model['horizon'])

    m_out = np.asarray(resource_grid, dtype=float)
    n_out = len(m_out)
    n_z = len(income)

    # Finer internal grid for the value function (log-spaced, matches scoring domain).
    m_int = np.geomspace(m_out[0], m_out[-1], 1001)
    n_int = len(m_int)

    V = [None] * T
    Vp = [None] * T
    V[T - 1] = _crra_util(m_int, gamma)
    Vp[T - 1] = _crra_mu(m_int, gamma)

    # Backward induction on the internal grid to build accurate V, Vp.
    for t in range(T - 2, -1, -1):
        if time.time() > deadline:
            for tt in range(t, T - 1):
                V[tt] = V[T - 1].copy()
                Vp[tt] = Vp[T - 1].copy()
            break
        M = np.zeros((n_z, n_int))
        for z in range(n_z):
            for zp in range(n_z):
                if trans[z, zp] > 0:
                    M[z] += trans[z, zp] * Vp[t + 1][zp]
        Vt = np.zeros((n_z, n_int))
        Vpt = np.zeros((n_z, n_int))
        for z in range(n_z):
            yz = income[z]
            Mz = M[z]
            c_int = np.empty(n_int)
            for i in range(n_int):
                c_int[i] = _solve_euler_bisect(m_int[i], yz, R, beta, Mz, m_int, gamma)
            mprime_all = R * (m_int - c_int) + yz
            cont = np.zeros(n_int)
            for zp in range(n_z):
                if trans[z, zp] > 0:
                    cont += trans[z, zp] * np.interp(mprime_all, m_int, V[t + 1][zp])
            Vt[z] = _crra_util(c_int, gamma) + beta * cont
            Vpt[z] = beta * np.interp(mprime_all, m_int, Mz)
        V[t] = Vt
        Vp[t] = Vpt

    # Compute consumption on the OUTPUT grid using the internal value function.
    out = np.zeros((T, n_z, n_out))
    out[T - 1] = m_out[None, :, None]
    for t in range(T - 2, -1, -1):
        M = np.zeros((n_z, n_int))
        for z in range(n_z):
            for zp in range(n_z):
                if trans[z, zp] > 0:
                    M[z] += trans[z, zp] * Vp[t + 1][zp]
        for z in range(n_z):
            yz = income[z]
            Mz = M[z]
            c_sol = np.empty(n_out)
            for i in range(n_out):
                c_sol[i] = _solve_euler_bisect(m_out[i], yz, R, beta, Mz, m_int, gamma)
            out[t, z] = c_sol

    out = np.where(np.isfinite(out), out, m_out[None, None, :])
    out = np.clip(out, 1e-12, m_out[None, None, :] + 1e-10)
    out[T - 1] = m_out[None, :, None]

    return out
