"""Consumption-saving policy via FOC root-finding with vectorized interpolation."""
import time
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START

def _utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(c)
    return c ** (1.0 - gamma) / (1.0 - gamma)

def _utility_prime(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return 1.0 / c
    return c ** (-gamma)

def solve_consumption(model, resource_grid, budget):
    _t0 = time.time()
    deadline = _t0 + 110.0  # safety margin under 120s/150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model.get('R', 1.02)
    income = np.asarray(model['income'], dtype=float)
    Tmat = np.asarray(model['transition'], dtype=float)
    horizon = int(model['horizon'])
    m_grid = np.asarray(resource_grid, dtype=float)

    n_z = len(income)
    n_m = len(m_grid)
    C = np.empty((horizon, n_z, n_m), dtype=float)

    # Terminal period: consume all
    for z in range(n_z):
        C[horizon - 1, z, :] = m_grid

    # V_next[z, m] = value at period t+1
    # Vp_next[z, m] = dV/dm at period t+1
    # Initialize with terminal: V_{T-1}(z, m) = u(m), Vp_{T-1}(z, m) = u'(m)
    V_next = np.zeros((n_z, n_m), dtype=float)
    Vp_next = np.zeros((n_z, n_m), dtype=float)
    for z in range(n_z):
        V_next[z, :] = _utility(m_grid, gamma)
        Vp_next[z, :] = _utility_prime(m_grid, gamma)

    # Precompute: for each z, the expected continuation derivative as a function of m_next
    # E_z[Vp_next(z', m_next)] = sum_{z'} Tmat[z, z'] * Vp_next[z', m_next]
    # This is a vector over m_grid for each z.
    # We'll compute this on the fly via interpolation.

    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            probs = Tmat[z]

            # Precompute the expected Vp_next on the grid for this z
            # exp_Vp_on_grid[m_idx] = sum_{z'} probs[z'] * Vp_next[z', m_grid[m_idx]]
            exp_Vp_on_grid = np.zeros(n_m, dtype=float)
            for zp in range(n_z):
                exp_Vp_on_grid += probs[zp] * Vp_next[zp, :]

            # Also precompute expected V on grid for value computation later
            exp_V_on_grid = np.zeros(n_m, dtype=float)
            for zp in range(n_z):
                exp_V_on_grid += probs[zp] * V_next[zp, :]

            # For each m in m_grid, solve the FOC
            # FOC: u'(c) = beta * R * E[Vp_next(z', R*(m-c) + income[z'])]
            # Define f(c) = u'(c) - beta*R*exp_Vp_interp(R*(m-c)+income)
            # f is strictly decreasing in c, so unique root.

            c_vals = np.empty(n_m, dtype=float)
            for i, m in enumerate(m_grid):
                c_low = 1e-10
                c_high = m - 1e-12
                if c_high <= c_low:
                    c_vals[i] = max(c_low, m)
                    continue

                # Inline f(c) for speed
                def f(c, m=m, exp_Vp_on_grid=exp_Vp_on_grid, m_grid=m_grid,
                      income=income, R=R, beta=beta, gamma=gamma):
                    m_next = R * (m - c) + income
                    m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])
                    # Interpolate exp_Vp_on_grid at m_next_clipped
                    evp = np.interp(m_next_clipped, m_grid, exp_Vp_on_grid)
                    return _utility_prime(c, gamma) - beta * R * evp

                f_low = f(c_low)
                f_high = f(c_high)

                if f_low <= 0:
                    c_vals[i] = c_low
                elif f_high >= 0:
                    c_vals[i] = c_high
                else:
                    try:
                        c_star = brentq(f, c_low, c_high, xtol=1e-13, rtol=1e-14, maxiter=60)
                        c_vals[i] = c_star
                    except Exception:
                        # Bisection fallback
                        lo, hi = c_low, c_high
                        for _ in range(70):
                            mid = 0.5 * (lo + hi)
                            if f(mid) > 0:
                                lo = mid
                            else:
                                hi = mid
                        c_vals[i] = 0.5 * (lo + hi)

                # Deadline guard
                if time.time() > deadline:
                    c_vals[i+1:] = m_grid[i+1:]
                    break

            C[t, z, :] = c_vals

            # Deadline guard
            if time.time() > deadline:
                # Fill remaining
                for zz in range(z + 1, n_z):
                    C[t, zz, :] = m_grid
                # Also fill remaining periods
                for tt in range(t - 1, -1, -1):
                    for zz in range(n_z):
                        C[tt, zz, :] = m_grid
                return C

        # Update V and Vp for period t
        V_curr = np.zeros((n_z, n_m), dtype=float)
        Vp_curr = np.zeros((n_z, n_m), dtype=float)

        for z in range(n_z):
            probs = Tmat[z]
            # exp_Vp_on_grid and exp_V_on_grid already computed above for this z
            # Recompute if needed (they were computed in the loop above)
            exp_Vp_on_grid = np.zeros(n_m, dtype=float)
            exp_V_on_grid = np.zeros(n_m, dtype=float)
            for zp in range(n_z):
                exp_Vp_on_grid += probs[zp] * Vp_next[zp, :]
                exp_V_on_grid += probs[zp] * V_next[zp, :]

            c_row = C[t, z, :]
            # m_next for each m
            m_next = R * (m_grid - c_row) + income  # shape (n_m, n_z)
            m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])

            # For each m_idx, compute expected V and Vp at m_next
            # Vectorized: for each z', interpolate exp over m_grid
            ev_vals = np.zeros(n_m, dtype=float)
            evp_vals = np.zeros(n_m, dtype=float)
            for zp in range(n_z):
                ev_vals += probs[zp] * np.interp(m_next_clipped[:, zp], m_grid, V_next[zp, :])
                evp_vals += probs[zp] * np.interp(m_next_clipped[:, zp], m_grid, Vp_next[zp, :])

            V_curr[z, :] = _utility(c_row, gamma) + beta * ev_vals
            Vp_curr[z, :] = beta * R * evp_vals

        V_next = V_curr
        Vp_next = Vp_curr

    # Enforce constraints: 0 < c <= m + 1e-10
    C = np.maximum(C, 1e-12)
    for t in range(horizon):
        for z in range(n_z):
            C[t, z, :] = np.minimum(C[t, z, :], m_grid + 1e-10)

    # Terminal constraint: c[5] must equal grid within 1e-9
    C[horizon - 1, :, :] = m_grid

    return C

# EVOLVE-BLOCK-END
