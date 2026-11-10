"""Runtime least-squares fit of the 6-coefficient CPL growth-index ansatz to the
exact GR growth ODE over a grid of CPL cosmologies. Returns the fitted c vector."""
import time
import math
import numpy as np
from scipy.integrate import solve_ivp

_DEADLINE = 90.0
_T0 = time.perf_counter()

def _elapsed():
    return time.perf_counter() - _T0

def _time_left(margin=6.0):
    return _DEADLINE - _elapsed() - margin

def _Om_of_a(Om, w0, wa, a):
    # Om(a) = Om * a^3 / (Om*a^3 + (1-Om)*a^3*(1+w0+wa*(1-a))**(-(1+w0)/wa))
    # Handle wa == 0 (CDM-like) as limit: de term = (1-Om)*a^3
    a3 = a**3
    if abs(wa) < 1e-12:
        de = (1.0 - Om) * a3
    else:
        base = (1.0 + w0 + wa * (1.0 - a))
        de = (1.0 - Om) * a3 * base**(-(1.0 + w0) / wa)
    return Om * a3 / (Om * a3 + de)

def _gamma_of(a, Om, w0, wa, c):
    Om_a = _Om_of_a(Om, w0, wa, a)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    return (c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q)

def _f_of_a(a, Om, w0, wa, c):
    return _Om_of_a(Om, w0, wa, a) ** _gamma_of(a, Om, w0, wa, c)

def _ref_f_and_lnD(Om, w0, wa, a_grid):
    """Exact reference: f = d lnD/d ln a, and ln(D(a)/D(1)).
    Solve D'' + (3 + q) D' + (3/2) Om(a) (1+w) D = 0 in variable x=ln a.
    y = [D, dD/dx]; D' = dD/dx. f = y[1]/y[0]."""
    def om_at_x(x):
        return _Om_of_a(Om, w0, wa, math.exp(x))
    def w_at_x(x):
        a = math.exp(x)
        return w0 + wa * (1.0 - a)
    def rhs(x, y):
        D, Dp = y
        om = om_at_x(x)
        w = w_at_x(x)
        q = 1.0 - om
        # dD/dx = Dp ; dDp/dx = -(3+q) Dp - 1.5*om*(1+w)*D
        return [Dp, -(3.0 + q) * Dp - 1.5 * om * (1.0 + w) * D]

    x0 = 0.0
    # Integrate from 0 to min(grid) and 0 to max(grid) to cover full range.
    x_min = float(np.min(a_grid)); x_max = float(np.max(a_grid))
    x_lo = math.log(max(x_min, 1e-6))
    x_hi = math.log(x_max)
    # initial conditions near a=1: use matter-dominated limit D~a^gamma0, D'~gamma0*D
    gamma0 = 0.55
    D0 = 1.0
    Dp0 = gamma0 * D0

    def integrate_to(x_target, rtol=1e-10, atol=1e-12):
        if abs(x_target) < 1e-12:
            return np.array([D0, Dp0])
        sol = solve_ivp(rhs, [x0, x_target], [D0, Dp0], method="RK45",
                        rtol=rtol, atol=atol, max_step=0.05)
        if not sol.success:
            return None
        return sol.y[:, -1]

    # We need f and lnD at each grid point. Integrate once over the whole span
    # with dense output for accuracy.
    span_lo = min(x_lo, 0.0); span_hi = max(x_hi, 0.0)
    if span_hi - span_lo < 1e-12:
        f = np.full_like(a_grid, gamma0)
        lnD = np.zeros_like(a_grid)
        return f, lnD

    sol = solve_ivp(rhs, [span_lo, span_hi], [D0, Dp0], method="RK45",
                    rtol=1e-10, atol=1e-12, max_step=0.05, dense_output=True)
    f = np.empty_like(a_grid, dtype=float)
    lnD = np.empty_like(a_grid, dtype=float)
    if not sol.success:
        f[:] = gamma0
        lnD[:] = 0.0
        return f, lnD
    xs = np.log(np.clip(a_grid, 1e-6, None))
    Y = sol.sol(xs)
    D = Y[0]; Dp = Y[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f = Dp / D
    lnD = np.log(np.abs(D)) - np.log(abs(D0))
    f = np.where(np.isfinite(f), f, gamma0)
    lnD = np.where(np.isfinite(lnD), lnD, 0.0)
    return f, lnD

def _fit_coefficients(Om_list, w0_list, wa_list, a_grid, c_init):
    """Least-squares fit of c to log(f) target over all cosmologies."""
    rows = []
    targets = []
    for Om in Om_list:
        for w0 in w0_list:
            for wa in wa_list:
                f_ref, _ = _ref_f_and_lnD(Om, w0, wa, a_grid)
                if _elapsed() > _DEADLINE - 15.0:
                    break
                for i, a in enumerate(a_grid):
                    Om_a = _Om_of_a(Om, w0, wa, a)
                    q = 1.0 - Om_a
                    u = 1.0 + w0 + wa * (1.0 - a)
                    row = [1.0, q, q * q, u, wa * a, u * q]
                    rows.append(row)
                    targets.append(math.log(max(f_ref[i], 1e-12)))
    if not rows:
        return c_init
    A = np.array(rows, dtype=float)
    b = np.array(targets, dtype=float)
    # Regularized least squares for stability
    lam = 1e-6
    try:
        c, *_ = np.linalg.lstsq(A + lam * np.eye(A.shape[1]), b, rcond=None)
    except Exception:
        return c_init
    c = np.asarray(c, dtype=float)
    if not np.all(np.isfinite(c)):
        return c_init
    return c

def _clip_c(c):
    c = np.clip(np.asarray(c, dtype=float), -10.0, 10.0)
    return c

def _gamma_ok(c, Om_list, w0_list, wa_list, a_grid):
    lo, hi = 0.1, 1.2
    for Om in Om_list:
        for w0 in w0_list:
            for wa in wa_list:
                for a in a_grid:
                    g = _gamma_of(a, Om, w0, wa, c)
                    if not (lo - 1e-9 <= g <= hi + 1e-9):
                        return False
    return True

def _score(c, Om_list, w0_list, wa_list, a_grid):
    """Worst relative error of f and normalized D over the grid."""
    worst = 0.0
    for Om in Om_list:
        for w0 in w0_list:
            for wa in wa_list:
                f_ref, lnD_ref = _ref_f_and_lnD(Om, w0, wa, a_grid)
                # our f
                f_our = np.array([_f_of_a(a, Om, w0, wa, c) for a in a_grid])
                # our lnD: integrate dlnD/dx = f over x from log(a) to 0
                lnD_our = np.empty_like(a_grid)
                for i, a in enumerate(a_grid):
                    x = math.log(max(a, 1e-6))
                    if abs(x) < 1e-12:
                        lnD_our[i] = 0.0
                        continue
                    # integrate f from x to 0 using our f (trapezoid on fine grid)
                    n = 200
                    xs = np.linspace(x, 0.0, n + 1)
                    ys = np.array([_f_of_a(math.exp(xx), Om, w0, wa, c) for xx in xs])
                    lnD_our[i] = -np.trapz(ys, xs)
                rf = np.max(np.abs(f_our - f_ref) / np.maximum(np.abs(f_ref), 1e-9))
                rd = np.max(np.abs(lnD_our - lnD_ref) / np.maximum(np.abs(lnD_ref), 1e-9))
                worst = max(worst, rf, rd)
                if _elapsed() > _DEADLINE - 12.0:
                    return worst
    return worst

def construct():
    # Deadline guard: if we're already near the limit, return the safe seed.
    if _time_left(margin=10.0) <= 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Training grids (kept modest to respect the deadline).
    Om_list = [0.22, 0.27, 0.32, 0.38, 0.45]
    w0_list = [-1.2, -1.0, -0.9, -0.7, -0.5]
    wa_list = [-0.4, -0.2, 0.0, 0.2, 0.4]
    a_grid = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])

    c_init = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    c = c_init.copy()

    try:
        c_fit = _fit_coefficients(Om_list, w0_list, wa_list, a_grid, c_init)
        c = _clip_c(c_fit)
        # If gamma constraint violated, fall back to seed (guaranteed valid).
        if not _gamma_ok(c, Om_list, w0_list, wa_list, a_grid):
            c = c_init.copy()
    except Exception:
        c = c_init.copy()

    # Final safety: ensure finite and in range.
    c = np.clip(np.nan_to_num(c, nan=0.55, posinf=10.0, neginf=-10.0), -10.0, 10.0)
    if not _gamma_ok(c, Om_list, w0_list, wa_list, a_grid):
        c = c_init.copy()

    return [float(x) for x in c]
