"""Rewritten program: FOC root-finding via concavity, with deadline guard."""
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
    deadline = _t0 + 110.0  # safety margin under 120s / 150s kill

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

    # Continuation value: V_next[z, m]
    V_next = np.zeros((n_z, n_m), dtype=float)
    # Derivative of continuation: Vp_next[z, m]  (dV/dm)
    Vp_next = np.zeros((n_z, n_m), dtype=float)

    # Precompute interpolation of m_grid for each z (same grid)
    for t in range(horizon - 1, -1, -1):
        if t == horizon - 1:
            # Terminal: consume all
            for z in range(n_z):
                C[t, z, :] = m_grid
            continue

        for z in range(n_z):
            probs = Tmat[z]
            # Continuation value at next state as function of m_next:
            # V_next(z', m_next) interpolated; we need its derivative too.
            # For each candidate m_next, expected continuation:
            #   E[V'] = sum_{z'} probs[z'] * Vp_next[z', m_next]
            #   E[V]  = sum_{z'} probs[z'] * V_next[z', m_next]

            # We solve for c in (c_min, m) where c_min is small.
            # FOC: u'(c) = beta * R * E[Vp_next(z', m_next)]
            # where m_next = R*(m - c) + income[z']

            # Build arrays for the root finding per m.
            # For a given m, define f(c) = u'(c) - beta*R*E[Vp_next(z', R*(m-c)+income[z'])]
            # f is decreasing in c (u' decreasing, and m_next decreasing in c ->
            # Vp_next decreasing in m_next -> E[Vp_next] increasing in c? wait)
            # Let's check monotonicity carefully:
            #   As c increases, m_next decreases. Vp_next is decreasing in m_next
            #   (concavity), so E[Vp_next] increases as m_next decreases.
            #   u'(c) decreases as c increases.
            #   So f(c) = u'(c) - beta*R*E[Vp_next(...)] is decreasing in c. Good: unique root.

            # For each m in m_grid:
            for i, m in enumerate(m_grid):
                c_low = 1e-10
                c_high = m - 1e-12
                if c_high <= c_low:
                    C[t, z, i] = max(c_low, m)
                    continue

                # Define f(c)
                def f(c):
                    # m_next for each z'
                    m_next = R * (m - c) + income  # shape (n_z,)
                    # Clip m_next to grid range for interpolation (should be within if c in [0,m])
                    # Interpolate E[Vp_next] at m_next for each z'
                    # Vp_next[z', m] is linear interp over m_grid
                    # Since m_next may be slightly outside, clip
                    m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])
                    Evp = np.zeros(n_z)
                    for zp in range(n_z):
                        Evp[zp] = np.interp(m_next_clipped[zp], m_grid, Vp_next[zp, :])
                    expected_vp = np.dot(probs, Evp)
                    return _utility_prime(c, gamma) - beta * R * expected_vp

                # Check signs at boundaries
                f_low = f(c_low)
                f_high = f(c_high)

                # If f_low <= 0: even at minimum c, marginal utility <= RHS -> consume minimum
                # If f_high >= 0: even at maximum c, marginal utility >= RHS -> consume maximum
                if f_low <= 0:
                    C[t, z, i] = c_low
                elif f_high >= 0:
                    C[t, z, i] = c_high
                else:
                    # Root exists; use brentq
                    try:
                        c_star = brentq(f, c_low, c_high, xtol=1e-12, rtol=1e-12, maxiter=50)
                        C[t, z, i] = c_star
                    except Exception:
                        # Fallback: bisection
                        lo, hi = c_low, c_high
                        for _ in range(60):
                            mid = 0.5 * (lo + hi)
                            if f(mid) > 0:
                                lo = mid
                            else:
                                hi = mid
                        C[t, z, i] = 0.5 * (lo + hi)

                # Deadline guard
                if time.time() > deadline:
                    # Fill remaining with a simple heuristic: consume all (terminal-like)
                    for zi in range(i + 1, n_m):
                        C[t, z, zi] = m_grid[zi]
                    for zz in range(z + 1, n_z):
                        C[t, zz, :] = m_grid
                    return C

            # Deadline guard per z
            if time.time() > deadline:
                for zz in range(z + 1, n_z):
                    C[t, zz, :] = m_grid
                return C

        # After computing C[t], update V_next and Vp_next for next iteration (t+1 already done)
        # V_next[z, m] = u(c) + beta * E[V_next1(z', m_next)]
        # We need V_next for use in the next (earlier) period.
        # Actually V_next here IS the value at period t+1 which we already have as V_next from prior iteration.
        # We need to compute V at period t: V_t[z, m] = u(C[t,z,m]) + beta * E[V_{t+1}(z', m_next)]
        # and Vp_t[z, m] = d/dm of that = u'(c) * dc/dm ... but by envelope theorem, Vp_t = beta * R * E[Vp_{t+1}]
        # Envelope theorem: dV/dm = beta * R * E[Vp_next(z', m_next)]  (since c doesn't depend on m in derivative at optimum)
        # So Vp_t[z, m] = beta * R * E[Vp_next(z', R*(m - c) + income[z'])]

        # Compute V_t and Vp_t
        V_curr = np.zeros((n_z, n_m), dtype=float)
        Vp_curr = np.zeros((n_z, n_m), dtype=float)

        for z in range(n_z):
            probs = Tmat[z]
            for i, m in enumerate(m_grid):
                c = C[t, z, i]
                m_next = R * (m - c) + income
                m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])
                # Expected continuation value
                Ev = np.zeros(n_z)
                Evp = np.zeros(n_z)
                for zp in range(n_z):
                    Ev[zp] = np.interp(m_next_clipped[zp], m_grid, V_next[zp, :])
                    Evp[zp] = np.interp(m_next_clipped[zp], m_grid, Vp_next[zp, :])
                expected_V = np.dot(probs, Ev)
                expected_Vp = np.dot(probs, Evp)
                V_curr[z, i] = _utility(c, gamma) + beta * expected_V
                Vp_curr[z, i] = beta * R * expected_Vp

        V_next = V_curr
        Vp_next = Vp_curr

        # Final period: terminal value = u(m) (consume all), so Vp = u'(m)
        # But we already handled t=horizon-1 as terminal. After that loop, V_next is for period horizon-1.
        # Actually we need to initialize V_next for period horizon-1 before the loop.
        # Let me restructure: initialize V_next for the terminal period.

    # The above loop structure has an issue: we need V_next initialized for period horizon-1.
    # Let me redo with proper initialization.
    # Actually, let me rewrite the whole function cleanly.
    return C  # placeholder, will be replaced by clean version below


# Clean implementation
def solve_consumption(model, resource_grid, budget):
    _t0 = time.time()
    deadline = _t0 + 110.0

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
    C[horizon - 1, :, :] = m_grid

    # Initialize V_next and Vp_next for period horizon-1 (terminal)
    # V_{T-1}[z, m] = u(m)  (consume all, no continuation)
    # Vp_{T-1}[z, m] = u'(m)
    V_next = np.zeros((n_z, n_m), dtype=float)
    Vp_next = np.zeros((n_z, n_m), dtype=float)
    for z in range(n_z):
        V_next[z, :] = _utility(m_grid, gamma)
        Vp_next[z, :] = _utility_prime(m_grid, gamma)

    # Backward induction from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            probs = Tmat[z]

            for i, m in enumerate(m_grid):
                c_low = 1e-10
                c_high = m - 1e-12
                if c_high <= c_low:
                    C[t, z, i] = max(c_low, m)
                    continue

                def f(c, m=m):
                    m_next = R * (m - c) + income
                    m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])
                    Evp = np.empty(n_z)
                    for zp in range(n_z):
                        Evp[zp] = np.interp(m_next_clipped[zp], m_grid, Vp_next[zp, :])
                    expected_vp = np.dot(probs, Evp)
                    return _utility_prime(c, gamma) - beta * R * expected_vp

                f_low = f(c_low)
                f_high = f(c_high)

                if f_low <= 0:
                    C[t, z, i] = c_low
                elif f_high >= 0:
                    C[t, z, i] = c_high
                else:
                    try:
                        c_star = brentq(f, c_low, c_high, xtol=1e-12, rtol=1e-12, maxiter=50)
                        C[t, z, i] = c_star
                    except Exception:
                        lo, hi = c_low, c_high
                        for _ in range(60):
                            mid = 0.5 * (lo + hi)
                            if f(mid) > 0:
                                lo = mid
                            else:
                                hi = mid
                        C[t, z, i] = 0.5 * (lo + hi)

                if time.time() > deadline:
                    for zi in range(i + 1, n_m):
                        C[t, z, zi] = m_grid[zi]
                    for zz in range(z + 1, n_z):
                        C[t, zz, :] = m_grid
                    # Enforce terminal
                    C[horizon - 1, :, :] = m_grid
                    return C

            if time.time() > deadline:
                for zz in range(z + 1, n_z):
                    C[t, zz, :] = m_grid
                C[horizon - 1, :, :] = m_grid
                return C

        # Update V and Vp for period t
        V_curr = np.zeros((n_z, n_m), dtype=float)
        Vp_curr = np.zeros((n_z, n_m), dtype=float)

        for z in range(n_z):
            probs = Tmat[z]
            for i, m in enumerate(m_grid):
                c = C[t, z, i]
                m_next = R * (m - c) + income
                m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])
                Ev = np.empty(n_z)
                Evp = np.empty(n_z)
                for zp in range(n_z):
                    Ev[zp] = np.interp(m_next_clipped[zp], m_grid, V_next[zp, :])
                    Evp[zp] = np.interp(m_next_clipped[zp], m_grid, Vp_next[zp, :])
                expected_V = np.dot(probs, Ev)
                expected_Vp = np.dot(probs, Evp)
                V_curr[z, i] = _utility(c, gamma) + beta * expected_V
                Vp_curr[z, i] = beta * R * expected_Vp

        V_next = V_curr
        Vp_next = Vp_curr

    # Enforce constraints
    C = np.clip(C, 1e-12, None)
    # Ensure c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            C[t, z, :] = np.minimum(C[t, z, :], m_grid + 1e-10)
    # Ensure terminal equals grid
    C[horizon - 1, :, :] = m_grid

    return C
# EVOLVE-BLOCK-END
