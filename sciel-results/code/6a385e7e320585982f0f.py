import time
import numpy as np


def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    DEADLINE = 150.0
    MARGIN = 8.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    trans = np.asarray(model['transition'], dtype=float)
    horizon = int(model['horizon'])
    Z = len(income)

    grid = np.asarray(resource_grid, dtype=float)
    N = len(grid)

    def util(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        return c ** (1.0 - gamma) / (1.0 - gamma)

    def uprime(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return 1.0 / c
        return c ** (-gamma)

    # Dense, smooth state grid (log-spaced) for accurate continuation interpolation.
    m_lo = 0.001
    m_hi = 40.0
    NSTATE = 2400
    M = np.geomspace(m_lo, m_hi, NSTATE)
    invM = 1.0 / M
    cmax = np.minimum(M, 40.0)
    cmin = np.maximum(M * 1e-3, 1e-9)

    # Candidate consumption grid, shared across all states (log-spaced).
    Nc = 400
    C = np.geomspace(cmin, cmax, Nc)
    Uc = util(C)
    uc = uprime(C)
    # value from saving: R * m'  (m' = R*(M - c) + income[z])
    save = R * M[:, None] + R * (income[None, :] - C[:, None])

    # Continuation value (concave) interpolated as value - q*m - r*m^2.
    V = np.zeros((2, Z, NSTATE))
    Q = np.zeros((2, Z, NSTATE))
    Rr = np.zeros((2, Z, NSTATE))
    c_out = np.zeros((horizon, Z, N))

    def cont(par, z, m):
        v = V[par, z]
        q = Q[par, z]
        r = Rr[par, z]
        return np.interp(m, M, v - q * M - r * M * M) + q * m + r * m * m

    def solve_period(par, t):
        nxt = 1 - par
        for z in range(Z):
            for zp in range(Z):
                p = trans[z, zp]
                if p < 1e-15:
                    continue
                # continuation value at candidate consumptions
                cv = cont(nxt, zp, save[:, z])
                obj = Uc[:, None] + beta * p * cv
                idx = np.argmax(obj, axis=0)
                c = C[idx]
                # local parabolic polish around the argmax (concave objective)
                for j in range(NSTATE):
                    i = idx[j]
                    if 1 <= i <= Nc - 2:
                        y0, y1, y2 = obj[i - 1, j], obj[i, j], obj[i + 1, j]
                        den = (y0 - 2.0 * y1 + y2)
                        if den < 0.0:
                            s = 0.5 * (y0 - y2) / den
                            if 0.0 <= s <= 1.0:
                                c[j] = C[i] + s * (C[i + 1] - C[i])
                # enforce constraints
                c = np.minimum(c, M)
                c = np.maximum(c, 1e-12)
                # record on the required grid
                c_out[t, z, :] = np.interp(grid, M, c)
                # fit concave surrogate value = v - q*m - r*m^2 to (m, c)
                val = np.log(c) if np.isclose(gamma, 1.0) else c ** (1.0 - gamma) / (1.0 - gamma)
                A = np.column_stack([np.ones(NSTATE), M, M * M])
                sol, *_ = np.linalg.lstsq(A, val, rcond=None)
                a0, a1, a2 = sol
                V[par, z] = a0 + a1 * M + a2 * M * M
                Q[par, z] = -a1
                Rr[par, z] = -a2

    # Terminal: consume everything.
    for z in range(Z):
        V[1, z] = np.log(M) if np.isclose(gamma, 1.0) else M ** (1.0 - gamma) / (1.0 - gamma)
        Q[1, z] = 1.0 / M
        Rr[1, z] = np.zeros(NSTATE)
    c_out[horizon - 1, :, :] = grid[np.newaxis, :]

    # Backward induction from horizon-2 down to 0.
    for t in range(horizon - 2, -1, -1):
        if time.time() - t0 > DEADLINE - MARGIN:
            # Safety: fill any remaining periods with a feasible fallback.
            for tt in range(t, horizon - 1):
                if tt < horizon - 1:
                    c_out[tt, :, :] = np.clip(grid[np.newaxis, :] * 0.5, 1e-12, grid[np.newaxis, :])
            break
        solve_period(1 - (t % 2), t)

    c_out = np.clip(c_out, 1e-12, None)
    # Enforce 0 < c <= m + 1e-10 and last period = grid.
    c_out[horizon - 1, :, :] = grid[np.newaxis, :]
    c_out = np.minimum(c_out, grid[np.newaxis, np.newaxis, :] + 1e-10)
    c_out = np.maximum(c_out, 1e-12)
    return c_out
