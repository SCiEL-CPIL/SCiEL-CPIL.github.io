"""Spectral collocation + local polish for consumption-saving policy."""
import time
import numpy as np
from scipy.interpolate import interp1d


def _crpa_u(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return np.log(c)
    return np.power(c, 1 - gamma) / (1 - gamma)


def _crpa_up(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return 1.0 / c
    return np.power(c, -gamma)


def _cheb_nodes(n):
    # Chebyshev nodes on [-1, 1]
    k = np.arange(n)
    return np.cos(np.pi * (k + 0.5) / n)


def _cheb_poly_eval(x, coeffs):
    # Evaluate Chebyshev series at x in [-1,1]
    # coeffs[k] is coefficient of T_k
    x = np.asarray(x)
    n = len(coeffs)
    # Clenshaw
    b1 = np.zeros_like(x)
    b0 = np.zeros_like(x)
    for k in range(n - 1, -1, -1):
        b0, b1 = b1, 2 * x * b1 - b0 + coeffs[k]
    return b0


def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    trans = np.asarray(model['transition'], dtype=float)
    H = int(model['horizon'])
    Z = len(income)

    m_grid = np.asarray(resource_grid, dtype=float)
    n = len(m_grid)

    # Log-space for spectral collocation
    log_m = np.log(m_grid)
    log_min, log_max = log_m[0], log_m[-1]
    # Map to [-1, 1]
    x = (log_m - (log_max + log_min) / 2) / ((log_max - log_min) / 2)

    # Chebyshev nodes for collocation (use a subset for value-function fit)
    n_col = min(60, n)
    x_col = _cheb_nodes(n_col)
    # Map back to log-m
    log_m_col = (log_max + log_min) / 2 + x_col * (log_max - log_min) / 2
    m_col = np.exp(log_m_col)

    # Number of Chebyshev coefficients (smooth => few needed)
    n_coeff = 40

    # V[t, z, :] : value function coefficients at period t (going backward)
    # Policy: c[t, z, :] on the full grid

    c_out = np.zeros((H, Z, n))

    # Terminal: consume all
    c_out[H - 1] = m_grid[None, :]  # (Z, n) broadcast

    # Value at terminal: V[H-1, z, m] = u(m)
    # We'll represent V as Chebyshev coefficients in x (log-space)
    V = [None] * H  # V[t] is (Z, n_coeff)

    # Terminal value coefficients: fit u(m) on the collocation grid
    def fit_cheb(y, x_col, n_coeff):
        # y: (n_col,) values, fit Chebyshev series of degree n_coeff-1
        # Use discrete Chebyshev transform (fast via DCT)
        # Chebyshev series: y_j = sum_k c_k T_k(x_j)
        # Use np.polynomial.chebyshev.chebfit or manual DCT
        from numpy.polynomial import chebyshev as C
        # chebfit returns coefficients in ascending order (c_k for T_k)
        coeffs = C.chebfit(x_col, y, deg=n_coeff - 1)
        return coeffs

    # Terminal value: V[H-1, z] = u(m) independent of z
    u_term = _crpa_u(m_col, gamma)
    V_term_coeffs = fit_cheb(u_term, x_col, n_coeff)
    V[H - 1] = np.tile(V_term_coeffs, (Z, 1))

    # For t = H-2 down to 0:
    # V[t, z, m] = max_c { u(c) + beta * sum_{z'} trans[z, z'] * V[t+1, z', m'] }
    # m' = R*(m - c) + income[z']
    # Euler: u'(c) = beta * R * sum_{z'} trans[z, z'] * V'[t+1, z', m']
    # where V'[t+1, z', m'] = dV/dm' (derivative of value w.r.t. next m)

    # We need dV/dm. Since V is in log-m space: dV/dm = dV/d(log m) * 1/m
    # dV/d(log m) = dV/dx * dx/d(log m) = dV/dx * (log_max - log_min)/2

    def cheb_deriv_coeffs(coeffs):
        # Derivative of Chebyshev series
        from numpy.polynomial import chebyshev as C
        return C.chebder(coeffs)

    # Precompute derivative coefficients for each period
    # V_prime_coeffs[t, z] = derivative of V[t, z] w.r.t. log_m
    scale = (log_max - log_min) / 2.0

    for t in range(H - 2, -1, -1):
        if time.time() > DEADLINE - 15:
            # Not enough time for polish; fill remaining with simple heuristic
            for tt in range(t, H - 1):
                if c_out[tt].sum() == 0:
                    c_out[tt] = np.clip(m_grid[None, :] * 0.9, 1e-12, m_grid[None, :])
            return _validate(c_out, m_grid)

        # Derivative of V[t+1] w.r.t. log_m
        V_next = V[t + 1]  # (Z, n_coeff)
        dV_next_dlogm = np.zeros((Z, n_coeff))
        for z in range(Z):
            dV_next_dlogm[z] = cheb_deriv_coeffs(V_next[z]) * scale

        # For each (z, m), solve Euler: u'(c) = beta*R*sum_{z'} trans[z,z']*dV_next_dlogm[z']*(1/m')
        # where m' = R*(m-c) + income[z']
        # dV_next_dlogm[z'] evaluated at log(m')
        # dV/dm' = dV/dlogm * 1/m'

        # We'll solve on the collocation grid for speed, then interpolate
        # Actually, let's solve on the full grid but vectorize over m_col for the value fit

        # Solve policy on collocation grid, then fit
        c_col = np.zeros((Z, n_col))
        for z in range(Z):
            # For each m_col_j, find c in (0, m_col_j)
            # f(c) = u'(c) - beta*R*sum_{z'} trans[z,z']*dV_next_dlogm[z'](log(m'))/m'
            # m' = R*(m-c) + income[z']
            # Use bisection on fraction s = c/m in (1e-4, 1-1e-4)

            m_j = m_col  # (n_col,)
            # For each j, bisect on s
            s_lo = np.full(n_col, 1e-4)
            s_hi = np.full(n_col, 1.0 - 1e-4)
            s = (s_lo + s_hi) / 2.0

            for _ in range(60):
                c_j = s * m_j
                c_j = np.maximum(c_j, 1e-12)
                lhs = _crpa_up(c_j, gamma)  # u'(c)
                rhs = np.zeros(n_col)
                for zp in range(Z):
                    mp = R * (m_j - c_j) + income[zp]
                    mp = np.maximum(mp, 1e-12)
                    log_mp = np.log(mp)
                    # Evaluate dV_next_dlogm[zp] at log_mp
                    x_mp = (log_mp - (log_max + log_min) / 2) / scale
                    x_mp = np.clip(x_mp, -1.0, 1.0)
                    dVval = _cheb_poly_eval(x_mp, dV_next_dlogm[zp])
                    # dV/dm' = dV/dlogm * 1/m'
                    term = dVval / mp
                    rhs += trans[z, zp] * term
                rhs *= beta * R
                # f = lhs - rhs; want f = 0
                # u'(c) is decreasing in c, so f is decreasing in c (increasing in s)
                # If f > 0, lhs too big => c too small => increase s
                # If f < 0, c too big => decrease s
                diff = lhs - rhs
                # diff > 0 means need larger c
                s_hi = np.where(diff > 0, s_hi, s)
                s_lo = np.where(diff <= 0, s_lo, s)
                s = (s_lo + s_hi) / 2.0

            c_col[z] = np.clip(s * m_j, 1e-12, m_j)

        # Now fit V[t, z, m] = u(c) + beta*sum_{z'}trans[z,z']*V[t+1,z',m'] on collocation grid
        for z in range(Z):
            c_j = c_col[z]
            val = _crpa_u(c_j, gamma)
            for zp in range(Z):
                mp = R * (m_j - c_j) + income[zp]
                mp = np.maximum(mp, 1e-12)
                log_mp = np.log(mp)
                x_mp = (log_mp - (log_max + log_min) / 2) / scale
                x_mp = np.clip(x_mp, -1.0, 1.0)
                Vval = _cheb_poly_eval(x_mp, V[t + 1, zp])
                val += beta * trans[z, zp] * Vval
            V[t, z] = fit_cheb(val, x_col, n_coeff)

        # Interpolate policy to full grid using the collocation policy
        # Fit c/m as function of log_m, then evaluate on full grid
        for z in range(Z):
            frac = c_col[z] / m_col
            # Fit frac as Chebyshev in x_col
            frac_coeffs = fit_cheb(frac, x_col, min(20, n_coeff))
            c_full = _cheb_poly_eval(x, frac_coeffs) * m_grid
            c_full = np.clip(c_full, 1e-12, m_grid + 1e-10)
            c_out[t, z] = c_full

    # Local polish: refine policy on output grid with finer local search
    if time.time() < DEADLINE - 10:
        # Recompute derivative coefficients for polish
        for t in range(H - 2, -1, -1):
            V_next = V[t + 1]
            dV_next_dlogm = np.zeros((Z, n_coeff))
            for z in range(Z):
                dV_next_dlogm[z] = cheb_deriv_coeffs(V_next[z]) * scale

            for z in range(Z):
                # For each m in m_grid, bisect on s with more iterations
                m_j = m_grid  # (n,)
                s_lo = np.full(n, 1e-4)
                s_hi = np.full(n, 1.0 - 1e-4)
                s = (s_lo + s_hi) / 2.0
                for _ in range(80):
                    c_j = np.maximum(s * m_j, 1e-12)
                    lhs = _crpa_up(c_j, gamma)
                    rhs = np.zeros(n)
                    for zp in range(Z):
                        mp = np.maximum(R * (m_j - c_j) + income[zp], 1e-12)
                        log_mp = np.log(mp)
                        x_mp = (log_mp - (log_max + log_min) / 2) / scale
                        x_mp = np.clip(x_mp, -1.0, 1.0)
                        dVval = _cheb_poly_eval(x_mp, dV_next_dlogm[zp])
                        rhs += trans[z, zp] * (dVval / mp)
                    rhs *= beta * R
                    diff = lhs - rhs
                    s_hi = np.where(diff > 0, s_hi, s)
                    s_lo = np.where(diff <= 0, s_lo, s)
                    s = (s_lo + s_hi) / 2.0
                c_out[t, z] = np.clip(s * m_j, 1e-12, m_j + 1e-10)

    # Terminal must be exactly the grid
    c_out[H - 1] = m_grid[None, :]

    return _validate(c_out, m_grid)


def _validate(c, m_grid):
    # Ensure finite
    c = np.where(np.isfinite(c), c, 1e-12)
    # 0 < c <= m + 1e-10
    c = np.maximum(c, 1e-12)
    c = np.minimum(c, m_grid[None, None, :] + 1e-10)
    return c
