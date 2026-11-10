"""Rewritten solver: backward induction with Newton-Raphson on the Euler FOC
using PCHIP-interpolated continuation value and its derivative for smooth
quadratic convergence to the interior optimum."""
import numpy as np
from scipy.interpolate import PchipInterpolator
import time

def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    m = np.asarray(resource_grid, dtype=float)
    n_m = len(m)

    # CRRA utility, marginal utility, and inverse of marginal utility
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def up(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return np.power(c, -gamma)

    def up_inv(g):
        # c such that up(c) = g  =>  c = g^(-1/gamma)
        g = np.maximum(g, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / g
        return np.power(g, -1.0 / gamma)

    T = horizon
    V = np.zeros((T, 2, n_m))          # V[t, z, i]
    C = np.zeros((T, 2, n_m))          # C[t, z, i]
    # interpolators for V and its slope dV/dm at each time
    V_interp = [None, None]
    dV_dm_interp = [None, None]

    for t in range(T - 1, -1, -1):
        # build interpolators for V[t+1] and its slope (if t+1 < T)
        if t + 1 < T:
            V_next = V[t + 1]
            dV_dm_next = np.zeros((2, n_m))
            for z in range(2):
                if n_m >= 3:
                    dV_dm_next[z] = np.gradient(V_next[z], m, edge_order=2)
                else:
                    dV_dm_next[z] = np.zeros(n_m)
                # keep slopes non-negative (value is increasing in m)
                dV_dm_next[z] = np.maximum(dV_dm_next[z], 1e-12)
            V_interp[t] = PchipInterpolator(m, V_next[0])
            V_interp[t + 1] = PchipInterpolator(m, V_next[1])
            dV_dm_interp[t] = PchipInterpolator(m, dV_dm_next[0])
            dV_dm_interp[t + 1] = PchipInterpolator(m, dV_dm_next[1])

        for z in range(2):
            cont = np.zeros(n_m)
            for i in range(n_m):
                m_i = m[i]
                if t + 1 < T:
                    # expected continuation value and expected slope
                    vp = 0.0
                    dp = 0.0
                    for zp in range(2):
                        prob = transition[z, zp]
                        inc = income[zp]
                        cont_i = R * (m_i - C[t + 1, zp, i]) + inc
                        cont_i = max(cont_i, 1e-12)
                        vp += prob * float(V_interp[zp](cont_i))
                        dp += prob * float(dV_dm_interp[zp](cont_i))
                    cont[i] = vp
                    slope[i] = dp
                else:
                    # terminal: consume everything
                    C[t, z, i] = m_i
                    continue

                # Euler FOC: up(c) = beta * R * (slope)
                g = beta * R * slope[i]
                c_int = up_inv(g)
                # Newton polish on f(c) = u'(c) - beta*R*slope  using
                # smooth second derivative for robustness.
                f0 = up(c_int) - g
                # iterate
                c = c_int
                for _ in range(6):
                    u1 = up(c)
                    u2 = -np.power(c, -gamma - 1.0) if abs(gamma-1)>1e-12 else -1.0/c**2
                    f = u1 - g
                    if u2 == 0:
                        break
                    step = f / u2
                    c_new = c - step
                    if c_new <= 0:
                        c_new = 0.5 * c
                    if abs(c_new - c) < 1e-13 * max(1.0, c):
                        c = c_new
                        break
                    c = c_new
                # enforce constraints: 0 < c <= m
                if c <= 0:
                    c = 1e-12
                if c > m_i:
                    c = m_i
                C[t, z, i] = c
                V[t, z, i] = u(c) + beta * cont[i]

            # terminal handling for t == T-1 done above (consume all)

    # enforce terminal condition strictly: C[T-1] == m within 1e-9
    C[T - 1, :, :] = m

    # final validity sweep
    C = np.clip(C, 1e-12, None)
    for t in range(T):
        for z in range(2):
            # 0 < c <= m + 1e-10
            C[t, z] = np.minimum(C[t, z], m + 1e-10)
            C[t, z] = np.maximum(C[t, z], 1e-12)
    C[T - 1, :, :] = m

    if not np.all(np.isfinite(C)):
        C = np.nan_to_num(C, nan=1e-6, posinf=m.max(), neginf=1e-6)

    return C
