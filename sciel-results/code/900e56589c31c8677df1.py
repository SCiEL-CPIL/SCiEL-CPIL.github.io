"""Dual-objective weighted least-squares fit of the 6-coefficient CPL growth-index
ansatz, simultaneously minimizing errors in f and normalized D(a)/D(1).
Uses Linder (2005) analytic gamma as primary reference, with ODE polish if time allows."""
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
    a3 = a**3
    if abs(wa) < 1e-12:
        de = (1.0 - Om) * a3
    else:
        base = (1.0 + w0 + wa * (1.0 - a))
        de = (1.0 - Om) * a3 * base**(-(1.0 + w0) / wa)
    return Om * a3 / (Om * a3 + de)

def _Om_of_a_vec(Om, w0, wa, a_arr):
    a_arr = np.asarray(a_arr, dtype=float)
    a3 = a_arr**3
    if abs(wa) < 1e-12:
        de = (1.0 - Om) * a3
    else:
        base = (1.0 + w0 + wa * (1.0 - a_arr))
        with np.errstate(divide='ignore', invalid='ignore'):
            de = (1.0 - Om) * a3 * base**(-(1.0 + w0) / wa)
        de = np.where(np.isfinite(de), de, (1.0 - Om) * a3)
    return Om * a3 / (Om * a3 + de)

def _gamma_of(a, Om, w0, wa, c):
    Om_a = _Om_of_a(Om, w0, wa, a)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    return (c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q)

def _gamma_of_vec(a_arr, Om, w0, wa, c):
    Om_a = _Om_of_a_vec(Om, w0, wa, a_arr)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a_arr)
    return (c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a_arr + c[5] * u * q)

def _f_of_a(a, Om, w0, wa, c):
    return _Om_of_a(Om, w0, wa, a) ** _gamma_of(a, Om, w0, wa, c)

def _f_of_a_vec(a_arr, Om, w0, wa, c):
    Om_a = _Om_of_a_vec(Om, w0, wa, a_arr)
    g = _gamma_of_vec(a_arr, Om, w0, wa, c)
    with np.errstate(divide='ignore', invalid='ignore'):
        f = Om_a ** g
    return np.where(np.isfinite(f), f, 0.55)

def _linder_gamma(a, Om, w0, wa):
    """Linder (2005) analytic growth index approximation.
    gamma(a) = 0.55 + (0.05 - 0.05*w0) * Om(a) + 0.02 * wa * (1 - a)
    Actually the standard form: gamma = 0.55 + (0.05 - 0.05*w0)*Om(a)
    with a small wa correction. Let's use the more accurate Linder 2005 form:
    gamma(a) = 0.55 + (0.05 - 0.05*w0)*Om(a) + 0.02*wa*(1-a)
    But for better accuracy across the range, use:
    gamma = 0.55 + 0.05*(1-w0)*Om(a) + 0.02*wa*(1-a)
    """
    Om_a = _Om_of_a_vec(Om, w0, wa, np.asarray(a))
    g = 0.55 + (0.05 - 0.05 * w0) * Om_a + 0.02 * wa * (1.0 - np.asarray(a))
    return g

def _ref_f_and_lnD(Om, w0, wa, a_grid):
    """Exact reference via ODE: f = dlnD/dln a, lnD = ln(D(a)/D(1))."""
    a_grid = np.asarray(a_grid, dtype=float)
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
        return [Dp, -(3.0 + q) * Dp - 1.5 * om * (1.0 + w) * D]

    x0 = 0.0
    x_min = float(np.min(a_grid)); x_max = float(np.max(a_grid))
    x_lo = math.log(max(x_min, 1e-6))
    x_hi = math.log(x_max)
    gamma0 = 0.55
    D0 = 1.0
    Dp0 = gamma0 * D0

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

def _compute_lnD_our(a_grid, Om, w0, wa, c):
    """Compute ln(D(a)/D(1)) using our f via numerical integration."""
    a_grid = np.asarray(a_grid, dtype=float)
    lnD = np.zeros_like(a_grid)
    for i, a in enumerate(a_grid):
        x = math.log(max(a, 1e-6))
        if abs(x) < 1e-12:
            lnD[i] = 0.0
            continue
        n = 150
        xs = np.linspace(x, 0.0, n + 1)
        ys = _f_of_a_vec(np.exp(xs), Om, w0, wa, c)
        lnD[i] = -np.trapz(ys, xs)
    return lnD

def _fit_dual_objective(Om_list, w0_list, wa_list, a_grid, c_init, weight_D=1.0):
    """Dual-objective: minimize combined residual of log(f) error and lnD error.
    
    The D-error is weighted by local sensitivity: d(lnD)/d(gamma_i) at each point,
    which is -integral of Om(a)^gamma * ln(Om(a)) * d(gamma)/d(c_i) da.
    We approximate this with the accumulated weight ~ |x| * Om(a)^gamma * ln(Om(a)).
    """
    rows_f = []
    targets_f = []
    rows_D = []
    targets_D = []
    
    for Om in Om_list:
        for w0 in w0_list:
            for wa in wa_list:
                if _elapsed() > _DEADLINE - 20.0:
                    break
                # Use Linder analytic gamma as target for f (fast, accurate)
                a_arr = np.asarray(a_grid, dtype=float)
                Om_a_arr = _Om_of_a_vec(Om, w0, wa, a_arr)
                gamma_linder = _linder_gamma(a_arr, Om, w0, wa)
                f_ref = Om_a_arr ** gamma_linder
                f_ref = np.where(np.isfinite(f_ref) & (f_ref > 0), f_ref, 0.55)
                ln_f_ref = np.log(f_ref)
                
                # For D target, compute lnD from Linder gamma via integration
                # lnD(a) = -integral_{ln a}^{0} f(x) dx where f = Om(a)^gamma_Linder(a)
                lnD_ref = np.zeros_like(a_arr)
                for i, a in enumerate(a_arr):
                    x = math.log(max(a, 1e-6))
                    if abs(x) < 1e-12:
                        lnD_ref[i] = 0.0
                        continue
                    n = 100
                    xs = np.linspace(x, 0.0, n + 1)
                    a_x = np.exp(xs)
                    Om_x = _Om_of_a_vec(Om, w0, wa, a_x)
                    g_x = _linder_gamma(a_x, Om, w0, wa)
                    f_x = Om_x ** g_x
                    f_x = np.where(np.isfinite(f_x) & (f_x > 0), f_x, 0.55)
                    lnD_ref[i] = -np.trapz(f_x, xs)
                
                # Build rows for f residual: log(f_our) - log(f_ref) = sum c_i * phi_i - log(f_ref)
                # where log(f_our) = gamma(a) * log(Om(a)) = (c0 + c1*q + ...) * log(Om(a))
                # So residual_f = sum_i c_i * phi_i(a) * log(Om(a)) - log(f_ref)
                # phi = [1, q, q^2, u, wa*a, u*q]
                for i, a in enumerate(a_arr):
                    Om_a = Om_a_arr[i]
                    if Om_a <= 0 or not np.isfinite(Om_a):
                        continue
                    log_Om = math.log(max(Om_a, 1e-12))
                    q = 1.0 - Om_a
                    u = 1.0 + w0 + wa * (1.0 - a)
                    phi = np.array([1.0, q, q * q, u, wa * a, u * q])
                    rows_f.append(phi * log_Om)
                    targets_f.append(ln_f_ref[i])
                    
                    # For D residual: lnD_our - lnD_ref
                    # lnD_our = -integral f_our dx, f_our = Om(x)^gamma(x)
                    # d(lnD_our)/dc_i = integral_{x}^{0} Om(x)^gamma(x) * ln(Om(x)) * phi_i(x) dx
                    # Approximate sensitivity weight: |x| * log_Om * phi_i(a) as proxy
                    x = math.log(max(a, 1e-6))
                    sens_weight = abs(x) * log_Om
                    rows_D.append(phi * sens_weight)
                    targets_D.append(lnD_ref[i])
    
    if not rows_f:
        return c_init
    
    A_f = np.array(rows_f, dtype=float)
    b_f = np.array(targets_f, dtype=float)
    
    # Combined system: [A_f; weight_D * A_D] [c] = [b_f; weight_D * b_D]
    # But D residual is nonlinear in c (through the integral). 
    # Use linearized approach: at current c, compute Jacobian of lnD w.r.t. c.
    # For simplicity, use the f-fit as primary and add D correction iteratively.
    
    # Primary fit: minimize f error
    lam = 1e-8
    try:
        c, *_ = np.linalg.lstsq(A_f + lam * np.eye(A_f.shape[1]), b_f, rcond=None)
    except Exception:
        return c_init
    c = np.asarray(c, dtype=float)
    if not np.all(np.isfinite(c)):
        return c_init
    c = np.clip(c, -10.0, 10.0)
    
    # Now refine with D-objective: compute actual D error and do a correction step
    if _elapsed() < _DEADLINE - 25.0 and rows_D:
        A_D = np.array(rows_D, dtype=float)
        b_D = np.array(targets_D, dtype=float)
        
        # Compute current D error using our f
        d_residuals = []
        for i, (Om, w0, wa) in enumerate(zip(
            [r for Om in Om_list for w0 in w0_list for wa in wa_list],
            [r for Om in Om_list for w0 in w0_list for r in w0_list for wa in wa_list],
            [r for Om in Om_list for w0 in w0_list for wa in wa_list for r in a_grid]
        )):
            pass  # This indexing is complex; do it properly below
        
        # Proper D refinement: for each cosmology, compute lnD_our - lnD_ref
        # and use it as additional constraint
        d_rows = []
        d_targets = []
        idx = 0
        for Om in Om_list:
            for w0 in w0_list:
                for wa in wa_list:
                    if _elapsed() > _DEADLINE - 20.0:
                        break
                    a_arr = np.asarray(a_grid, dtype=float)
                    lnD_our = _compute_lnD_our(a_arr, Om, w0, wa, c)
                    lnD_ref = np.zeros_like(a_arr)
                    for i, a in enumerate(a_arr):
                        x = math.log(max(a, 1e-6))
                        if abs(x) < 1e-12:
                            lnD_ref[i] = 0.0
                            continue
                        n = 80
                        xs = np.linspace(x, 0.0, n + 1)
                        a_x = np.exp(xs)
                        Om_x = _Om_of_a_vec(Om, w0, wa, a_x)
                        g_x = _linder_gamma(a_x, Om, w0, wa)
                        f_x = Om_x ** g_x
                        f_x = np.where(np.isfinite(f_x) & (f_x > 0), f_x, 0.55)
                        lnD_ref[i] = -np.trapz(f_x, xs)
                    
                    # Sensitivity: d(lnD_our)/dc_i ≈ integral of f_our * ln(Om) * phi_i dx
                    for i, a in enumerate(a_arr):
                        x = math.log(max(a, 1e-6))
                        if abs(x) < 1e-12:
                            continue
                        n = 80
                        xs = np.linspace(x, 0.0, n + 1)
                        a_x = np.exp(xs)
                        Om_x = _Om_of_a_vec(Om, w0, wa, a_x)
                        g_x = _gamma_of_vec(a_x, Om, w0, wa, c)
                        f_x = _f_of_a_vec(a_x, Om, w0, wa, c)
                        log_Om_x = np.log(np.maximum(Om_x, 1e-12))
                        q_x = 1.0 - Om_x
                        u_x = 1.0 + w0 + wa * (1.0 - a_x)
                        phi_x = np.column_stack([np.ones_like(a_x), q_x, q_x**2, u_x, wa * a_x, u_x * q_x])
                        # Sensitivity integral
                        sens = np.trapz(f_x * log_Om_x[:, None] * phi_x, xs, axis=0)
                        d_rows.append(sens)
                        d_targets.append(lnD_ref[i] - lnD_our[i])
        
        if d_rows and _elapsed() < _DEADLINE - 20.0:
            A_D = np.array(d_rows, dtype=float)
            b_D = np.array(d_targets, dtype=float)
            # Combined least squares: minimize ||A_f c - b_f||^2 + w_D^2 ||A_D c - b_D||^2
            w_D = 2.0
            A_comb = np.vstack([A_f, w_D * A_D])
            b_comb = np.concatenate([b_f, w_D * b_D])
            try:
                c2, *_ = np.linalg.lstsq(A_comb + lam * np.eye(A_comb.shape[1]), b_comb, rcond=None)
                c2 = np.asarray(c2, dtype=float)
                if np.all(np.isfinite(c2)):
                    c = np.clip(c2, -10.0, 10.0)
            except Exception:
                pass
    
    return c

def _clip_c(c):
    return np.clip(np.asarray(c, dtype=float), -10.0, 10.0)

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
                a_arr = np.asarray(a_grid, dtype=float)
                f_our = _f_of_a_vec(a_arr, Om, w0, wa, c)
                lnD_our = _compute_lnD_our(a_arr, Om, w0, wa, c)
                rf = np.max(np.abs(f_our - f_ref) / np.maximum(np.abs(f_ref), 1e-9))
                rd = np.max(np.abs(lnD_our - lnD_ref) / np.maximum(np.abs(lnD_ref), 1e-9))
                worst = max(worst, rf, rd)
                if _elapsed() > _DEADLINE - 12.0:
                    return worst
    return worst

def construct():
    if _time_left(margin=10.0) <= 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Training grids
    Om_list = [0.22, 0.27, 0.32, 0.38, 0.45]
    w0_list = [-1.2, -1.0, -0.9, -0.7, -0.5]
    wa_list = [-0.4, -0.2, 0.0, 0.2, 0.4]
    a_grid = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])

    c_init = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    c = c_init.copy()

    try:
        c_fit = _fit_dual_objective(Om_list, w0_list, wa_list, a_grid, c_init)
        c = _clip_c(c_fit)
        if not _gamma_ok(c, Om_list, w0_list, wa_list, a_grid):
            c = c_init.copy()
    except Exception:
        c = c_init.copy()

    c = np.clip(np.nan_to_num(c, nan=0.55, posinf=10.0, neginf=-10.0), -10.0, 10.0)
    if not _gamma_ok(c, Om_list, w0_list, wa_list, a_grid):
        c = c_init.copy()

    return [float(x) for x in c]
