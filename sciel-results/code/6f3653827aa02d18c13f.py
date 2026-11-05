"""Growth-index approximation for CPL dark-energy cosmologies.

Structural change: reparameterized minimax (Chebyshev) fit.

  1. Build reference f(a), D(a)/D(1) via the second-order growth ODE on a
     Chebyshev-spaced log(a) grid for a wider validation set.
  2. Reparameterize the 6 coefficients around a physically-motivated seed:
         c0 = 0.55 + d0
         c3 = -0.5  + d3
         c1, c2, c4, c5 = d1, d2, d4, d5
     so the optimizer only has to find small corrections.
  3. Stage A: weighted least-squares pre-fit (fast, gives a good start).
  4. Stage B: Chebyshev (minimax) polish via a linear program on the
     linearized residuals, directly targeting the max relative error.
  5. Return the optimized coefficients via construct().
"""

import numpy as np
from scipy.optimize import minimize, linprog
from scipy.integrate import solve_ivp
import time

_START = time.time()
_LIMIT = 82.0  # safety margin before 90 s kill


def _ok():
    return (time.time() - _START) < _LIMIT


# ── Physics ─────────────────────────────────────────────────────────────────

def _omega_m(a, Om0, w0, wa):
    """Ω_m(a) for CPL dark energy, a0 = 1."""
    a = np.asarray(a, dtype=float)
    a = np.clip(a, 1e-12, 1e6)
    a3 = a ** 3
    # E^2(a) = Om0 * a^-3 + (1-Om0) * a^{-3(1+w0+wa)} * exp(3*wa*(1-a))
    # Ω_m = Om0 * a^-3 / E^2
    de = (1.0 - Om0) * a3 ** (-(1.0 + w0 + wa)) * np.exp(3.0 * wa * (1.0 - a))
    return Om0 / (Om0 + de)


def _gamma(c, a, Om0, w0, wa):
    Om_m = _omega_m(a, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    c0, c1, c2, c3, c4, c5 = c
    return c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q


def _f_of(c, a, Om0, w0, wa):
    Om_m = np.clip(_omega_m(a, Om0, w0, wa), 1e-15, 1.0)
    return Om_m ** _gamma(c, a, Om0, w0, wa)


def _D_of(c, a, Om0, w0, wa):
    """D(a)/D(1) = exp(-∫_{ln a}^{0} f(e^t) dt)."""
    x = np.log(a)
    f = _f_of(c, a, Om0, w0, wa)
    # Cumulative trapezoid from x[0] to x[i]
    dx = np.diff(x)
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * dx)))
    J = cum[-1] - cum  # ∫_{x[i]}^{0} f dt
    return np.exp(-J)


# ── Reference ODE ───────────────────────────────────────────────────────────

def _reference(a_grid, Om0, w0, wa):
    x = np.log(a_grid)

    def ode(xv, y):
        a_v = np.exp(xv)
        Om_m = _omega_m(a_v, Om0, w0, wa)
        w = w0 + wa * (1.0 - a_v)
        dOm_dx = -3.0 * Om_m * (Om_m - 1.0)
        D, F = y
        Dpp = -(1.5 + (1.0 + w) - 1.5 * dOm_dx) * F + 1.5 * Om_m * D
        return [F, Dpp]

    a0 = a_grid[0]
    D0 = a0 ** (5.0 / 6.0)
    F0 = (5.0 / 6.0) * a0 ** (-1.0 / 6.0)

    sol = solve_ivp(ode, [x[0], x[-1]], [D0, F0],
                    t_eval=x, max_step=0.002, rtol=1e-10, atol=1e-12)
    D = sol.y[0]
    F = sol.y[1]
    D = D / D[-1]
    return F, D


# ── Validation set (wider than seed) ────────────────────────────────────────

_VALIDATION = [
    (0.30, -1.0, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.30, -0.5, 0.0),
    (0.30, -0.8, 0.3),
    (0.25, -1.0, 0.5),
    (0.35, -1.0, -0.3),
    (0.30, -1.2, 0.4),
    (0.30, -0.9, -0.4),
    (0.28, -1.0, 1.0),
    (0.30, -1.0, -1.0),
    (0.27, -0.9, 0.6),
    (0.32, -1.1, 0.2),
    (0.29, -0.7, -0.6),
]

# Chebyshev-spaced log(a) grid: better for L∞ error control
_N = 200
_x_lo = np.log(5e-4)
_x_hi = 0.0
_k = np.arange(_N)
_x_cheb = _x_hi - 0.5 * (_x_hi - _x_lo) * (1.0 - np.cos(np.pi * _k / (_N - 1)))
_EVAL_A = np.exp(_x_cheb)

_REF = []
for (Om0, w0, wa) in _VALIDATION:
    f_r, D_r = _reference(_EVAL_A, Om0, w0, wa)
    _REF.append((f_r, D_r, Om0, w0, wa))

# ── Reparameterization ──────────────────────────────────────────────────────
# c0 = 0.55 + d0,  c3 = -0.5 + d3
# c1, c2, c4, c5 = d1, d2, d4, d5
_SEED = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])


def _to_full(d):
    c = np.zeros(6)
    c[0] = _SEED[0] + d[0]
    c[1] = d[1]
    c[2] = d[2]
    c[3] = _SEED[3] + d[3]
    c[4] = d[4]
    c[5] = d[5]
    return c


def _to_delta(c):
    d = np.zeros(6)
    d[0] = c[0] - _SEED[0]
    d[1] = c[1]
    d[2] = c[2]
    d[3] = c[3] - _SEED[3]
    d[4] = c[4]
    d[5] = c[5]
    return d


# ── Objective (max relative error) ──────────────────────────────────────────

def _max_rel_err(c):
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6

    max_err = 0.0
    for f_r, D_r, Om0, w0, wa in _REF:
        gam = _gamma(c, _EVAL_A, Om0, w0, wa)
        if np.any(gam < 0.1) or np.any(gam > 1.2):
            viol = np.maximum(0.0, 0.1 - gam) + np.maximum(0.0, gam - 1.2)
            max_err = max(max_err, 5.0 * np.max(viol))
            continue

        f_m = _f_of(c, _EVAL_A, Om0, w0, wa)
        D_m = _D_of(c, _EVAL_A, Om0, w0, wa)

        rel_f = np.abs(f_m - f_r) / np.maximum(np.abs(f_r), 1e-12)
        rel_D = np.abs(D_m - D_r) / np.maximum(np.abs(D_r), 1e-12)

        max_err = max(max_err, np.max(rel_f), np.max(rel_D))

    return 100.0 * max_err


# ── Stage A: weighted least-squares pre-fit ─────────────────────────────────

def _lsq_pre_fit():
    """Weighted LSQ fit in the reparameterized space."""

    def _lsq_obj(d):
        c = _to_full(d)
        if np.any(c < -10) or np.any(c > 10):
            return 1e6
        sse = 0.0
        for f_r, D_r, Om0, w0, wa in _REF:
            gam = _gamma(c, _EVAL_A, Om0, w0, wa)
            if np.any(gam < 0.1) or np.any(gam > 1.2):
                return 1e6
            f_m = _f_of(c, _EVAL_A, Om0, w0, wa)
            D_m = _D_of(c, _EVAL_A, Om0, w0, wa)
            rf = (f_m - f_r) / np.maximum(np.abs(f_r), 1e-12)
            rD = (D_m - D_r) / np.maximum(np.abs(D_r), 1e-12)
            sse += np.sum(rf ** 2) + np.sum(rD ** 2)
        return sse

    d0 = np.zeros(6)
    try:
        res = minimize(_lsq_obj, d0, method='L-BFGS-B',
                       bounds=[(-1.0, 1.0)] * 6,
                       options={'maxiter': 300, 'ftol': 1e-14})
        if res.fun < _lsq_obj(d0):
            return res.x.copy()
    except Exception:
        pass
    return d0


# ── Stage B: Chebyshev minimax polish via linear programming ───────────────

def _chebyshev_polish(c_start, n_iter=3):
    """
    Iteratively solve a linear program that minimizes the max of
    linearized relative residuals.  Each iteration re-linearizes around
    the current c and solves:
        min  t
        s.t.  A @ d + b <= t
              -1 <= d_i <= 1   (small corrections)
    """
    c = np.array(c_start, dtype=float)
    for _ in range(n_iter):
        if not _ok():
            break

        # Build linearized residual rows
        A_rows = []
        b_rows = []

        for f_r, D_r, Om0, w0, wa in _REF:
            gam = _gamma(c, _EVAL_A, Om0, w0, wa)
            if np.any(gam < 0.1) or np.any(gam > 1.2):
                # Perturb to valid region
                c = np.clip(c, -10, 10)
                gam = _gamma(c, _EVAL_A, Om0, w0, wa)
                if np.any(gam < 0.1) or np.any(gam > 1.2):
                    continue

            f_m = _f_of(c, _EVAL_A, Om0, w0, wa)
            D_m = _D_of(c, _EVAL_A, Om0, w0, wa)

            # Linearize f: f ≈ f_m * (1 + dgamma * ln(Om_m))
            Om_m = np.clip(_omega_m(_EVAL_A, Om0, w0, wa), 1e-15, 1.0)
            lnOm = np.log(Om_m)

            # dgamma/dc_j for each coefficient
            q = 1.0 - Om_m
            u = 1.0 + w0 + wa * (1.0 - _EVAL_A)
            dgam_dc = np.zeros((6, len(_EVAL_A)))
            dgam_dc[0] = 1.0
            dgam_dc[1] = q
            dgam_dc[2] = 2.0 * q
            dgam_dc[3] = u
            dgam_dc[4] = wa * _EVAL_A
            dgam_dc[5] = u * q

            # f residual: (f_m*(1+dg*lnOm) - f_r) / f_r
            # = (f_m - f_r)/f_r + f_m*lnOm*dg / f_r
            for j in range(6):
                A_row = f_m * lnOm * dgam_dc[j] / np.maximum(np.abs(f_r), 1e-12)
                b_row = (f_m - f_r) / np.maximum(np.abs(f_r), 1e-12)
                A_rows.append(A_row)
                b_rows.append(b_row)
                A_rows.append(-A_row)
                b_rows.append(-b_row)

            # D residual: D ≈ D_m * (1 - ∫ delta_f dt)
            # delta_f = f_m * lnOm * dg
            # So D ≈ D_m - D_m * ∫_{ln a}^0 f_m*lnOm*dg dt
            # Linearize: (D_m - D_m*J_dg - D_r)/D_r
            # For simplicity, approximate J_dg per point:
            x = np.log(_EVAL_A)
            for j in range(6):
                delta_f = f_m * lnOm * dgam_dc[j]
                # ∫_{x_i}^{0} delta_f dt  (trapezoid)
                dx = np.diff(x)
                cum = np.concatenate(([0.0], np.cumsum(0.5 * (delta_f[1:] + delta_f[:-1]) * dx)))
                J_dg = cum[-1] - cum
                A_row = -D_m * J_dg / np.maximum(np.abs(D_r), 1e-12)
                b_row = (D_m - D_r) / np.maximum(np.abs(D_r), 1e-12)
                A_rows.append(A_row)
                b_rows.append(b_row)
                A_rows.append(-A_row)
                b_rows.append(-b_row)

        if not A_rows:
            break

        A = np.array(A_rows)
        b = np.array(b_rows)
        n = len(b_rows)
        m = 6

        # Variables: d[0..5], t
        # A @ d + b <= t  =>  A @ d - t <= -b
        A_lp = np.hstack([A, -np.ones((n, 1))])
        b_lp = -b

        # Bounds: -1 <= d_i <= 1, t >= 0
        bounds = [(-1.0, 1.0)] * 6 + [(0.0, None)]

        try:
            res = linprog(np.zeros(m + 1) + np.array([0, 0, 0, 0, 0, 0, 1.0]),
                          A_ub=A_lp, b_ub=b_lp, bounds=bounds, method='highs')
            if res.success:
                d = res.x[:6]
                c_new = _to_full(d)
                c_new = np.clip(c_new, -10, 10)
                # Accept if it actually improves
                if _max_rel_err(c_new) <= _max_rel_err(c) * 1.01:
                    c = c_new
                else:
                    break
            else:
                break
        except Exception:
            break

    return c


# ── Main optimization ───────────────────────────────────────────────────────

def _optimize():
    # Start from seed
    best_c = _SEED.copy()
    best_err = _max_rel_err(best_c)

    if not _ok():
        return best_c

    # Stage A: LSQ pre-fit
    try:
        d_lsq = _lsq_pre_fit()
        c_lsq = np.clip(_to_full(d_lsq), -10, 10)
        err_lsq = _max_rel_err(c_lsq)
        if err_lsq < best_err:
            best_c = c_lsq
            best_err = err_lsq
    except Exception:
        pass

    if not _ok():
        return best_c

    # Stage B: Chebyshev minimax polish
    try:
        c_cheb = _chebyshev_polish(best_c, n_iter=4)
        err_cheb = _max_rel_err(c_cheb)
        if err_cheb < best_err:
            best_c = c_cheb
            best_err = err_cheb
    except Exception:
        pass

    if not _ok():
        return best_c

    # Stage C: Nelder-Mead polish from the Chebyshev result
    try:
        res = minimize(lambda d: _max_rel_err(_to_full(d)),
                       _to_delta(best_c), method='Nelder-Mead',
                       options={'maxiter': 600, 'xatol': 1e-8, 'fatol': 1e-10,
                                'adaptive': True})
        c_nm = np.clip(_to_full(res.x), -10, 10)
        err_nm = _max_rel_err(c_nm)
        if err_nm < best_err:
            best_c = c_nm
            best_err = err_nm
    except Exception:
        pass

    if not _ok():
        return best_c

    # Stage D: A few restarts
    rng = np.random.default_rng(42)
    for i in range(6):
        if not _ok():
            break
        d_pert = rng.normal(0, 0.05, 6)
        d_start = _to_delta(best_c) + d_pert
        try:
            res = minimize(lambda d: _max_rel_err(_to_full(d)),
                           d_start, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10,
                                    'adaptive': True})
            c_r = np.clip(_to_full(res.x), -10, 10)
            err_r = _max_rel_err(c_r)
            if err_r < best_err:
                best_c = c_r
                best_err = err_r
        except Exception:
            pass

    return np.clip(best_c, -10, 10)


_COEFFS = _optimize()


def construct():
    return list(_COEFFS)
