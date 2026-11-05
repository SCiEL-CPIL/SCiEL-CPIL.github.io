"""Growth-index approximation for CPL dark-energy cosmologies.

Structural change: direct multi-start coordinate descent with adaptive grid.
Instead of anchor interpolation, we:
1. Pre-compute reference ODE solutions at module level.
2. Use a focused evaluation grid (denser at early times where gamma varies most).
3. Run multi-start coordinate descent directly on the 6 coefficients.
4. Each coordinate sweep is fast (vectorized), allowing many restarts.
"""

import numpy as np
from scipy.integrate import solve_ivp
import time

_START_TIME = time.time()
_TIME_LIMIT = 85.0

def _deadline_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

# ── Physics helpers ──────────────────────────────────────────────────────────

def _omega_m_of_a(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    a_safe = np.clip(a, 1e-12, 1e6)
    a_inv3 = a_safe ** (-3.0)
    de_factor = a_safe ** (-3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (a_safe - 1.0))
    Om_DE = (1.0 - Om0) * de_factor
    denom = Om0 * a_inv3 + Om_DE
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    Om_m = Om0 * a_inv3 / denom
    return Om_m

def _gamma_from_coeffs(c, a, Om0, w0, wa):
    c0, c1, c2, c3, c4, c5 = c
    Om_m = _omega_m_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _f_from_gamma(gamma, Om_m):
    Om_m_safe = np.clip(Om_m, 1e-15, 1.0)
    return Om_m_safe ** gamma

# ── Reference ODE solver ─────────────────────────────────────────────────────

def _growth_ode_reference(a_grid, Om0, w0, wa):
    a_min = a_grid[0]
    a_max = a_grid[-1]
    x_min = np.log(a_min)
    x_max = np.log(a_max)

    def ode(x, y):
        a = np.exp(x)
        Om_m = _omega_m_of_a(a, Om0, w0, wa)
        w = w0 + wa * (1.0 - a)
        dOm_m_dx = -3.0 * Om_m * (Om_m - 1.0)
        D_double = -(1.5 + (1.0 + w) - 1.5 * dOm_m_dx) * y[1] + 1.5 * Om_m * y[0]
        return [y[1], D_double]

    D0 = a_min ** (5.0/6.0)
    F0 = (5.0/6.0) * a_min ** (-1.0/6.0)

    try:
        sol = solve_ivp(ode, [x_min, x_max], [D0, F0],
                        t_eval=a_grid, max_step=0.005, rtol=1e-8, atol=1e-10)
        if not sol.success:
            raise RuntimeError("ODE integration failed")
        D_ref = sol.y[0] / sol.y[0][-1]
        F_ref = sol.y[1]
        return F_ref, D_ref
    except Exception:
        gamma_ref = 0.55
        Om_m_arr = _omega_m_of_a(a_grid, Om0, w0, wa)
        f_ref = Om_m_arr ** gamma_ref
        x = np.log(a_grid)
        dx = np.diff(x)
        f_avg = 0.5 * (f_ref[1:] + f_ref[:-1])
        cum_int = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
        integral_vals = cum_int - cum_int[-1]
        D_ref = np.exp(integral_vals)
        D_ref = D_ref / D_ref[-1]
        return f_ref, D_ref

# ── Evaluation grid: denser at early times ───────────────────────────────────

# Use log-spaced grid with more points at small a
_EVAL_A = np.concatenate([
    np.logspace(np.log10(1e-3), np.log10(0.01), 80),
    np.logspace(np.log10(0.01), np.log10(0.1), 60),
    np.logspace(np.log10(0.1), 0.0, 60)
])
_EVAL_A = np.unique(_EVAL_A)

# ── Validation cosmologies ───────────────────────────────────────────────────

_VALIDATION_COSMOLOGIES = [
    (0.30, -0.5, 0.0),
    (0.30, -1.0, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.30, -0.8, 0.3),
    (0.25, -1.0, 0.5),
    (0.35, -1.0, -0.3),
    (0.30, -1.2, 0.4),
    (0.30, -0.9, -0.4),
    (0.28, -1.0, 1.0),
]

# ── Pre-compute reference data at module level ───────────────────────────────

_REF_DATA = []

def _build_references():
    global _REF_DATA
    _REF_DATA = []
    for (Om0, w0, wa) in _VALIDATION_COSMOLOGIES:
        try:
            f_ref, D_ref = _growth_ode_reference(_EVAL_A, Om0, w0, wa)
            _REF_DATA.append((f_ref, D_ref, Om0, w0, wa))
        except Exception:
            _REF_DATA.append(None)

if _deadline_ok():
    _build_references()

# ── Objective function (vectorized) ──────────────────────────────────────────

def _objective_from_coeffs(c):
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6

    max_err = 0.0
    a_eval = _EVAL_A
    x_eval = np.log(a_eval)
    dx = np.diff(x_eval)
    
    for ref in _REF_DATA:
        if ref is None:
            continue
        f_ref, D_ref, Om0, w0, wa = ref

        Om_m_eval = _omega_m_of_a(a_eval, Om0, w0, wa)
        gamma_eval = _gamma_from_coeffs(c, a_eval, Om0, w0, wa)

        if np.any(gamma_eval < 0.1) or np.any(gamma_eval > 1.2):
            gamma_violation = np.maximum(0.0, 0.1 - gamma_eval) + np.maximum(0.0, gamma_eval - 1.2)
            max_err = max(max_err, 10.0 * np.max(gamma_violation))
            continue

        f_model = _f_from_gamma(gamma_eval, Om_m_eval)
        err_f = np.abs(f_model - f_ref) / np.abs(f_ref)
        max_err = max(max_err, np.max(err_f))

        # Compute D_model using trapezoidal integration in log(a)
        f_avg = 0.5 * (f_model[1:] + f_model[:-1])
        cum_int = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
        integral_vals = cum_int - cum_int[-1]
        D_model = np.exp(integral_vals)
        D_model = D_model / D_model[-1]

        err_D = np.abs(D_model - D_ref) / np.abs(D_ref)
        max_err = max(max_err, np.max(err_D))

        if not _deadline_ok():
            return 100.0 * max_err

    return 100.0 * max_err

# ── Coordinate descent with multi-start ──────────────────────────────────────

def _coordinate_descent(c_init, max_sweeps=200):
    """Perform coordinate descent on the 6 coefficients."""
    c = np.array(c_init, dtype=float)
    best_err = _objective_from_coeffs(c)
    
    for sweep in range(max_sweeps):
        if not _deadline_ok():
            break
        
        improved = False
        for i in range(6):
            # Try a few perturbations along coordinate i
            best_val = c[i]
            best_err_i = best_err
            
            # Adaptive step size
            step_sizes = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
            for step in step_sizes:
                for direction in [-1, 1]:
                    c_try = c.copy()
                    c_try[i] = c[i] + direction * step
                    if c_try[i] < -10 or c_try[i] > 10:
                        continue
                    err = _objective_from_coeffs(c_try)
                    if err < best_err_i:
                        best_err_i = err
                        best_val = c_try[i]
            
            if best_val != c[i]:
                c[i] = best_val
                best_err = best_err_i
                improved = True
        
        if not improved:
            break
    
    return c, best_err

# ── Construct ────────────────────────────────────────────────────────────────

def construct():
    if not _deadline_ok():
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    if not _REF_DATA:
        _build_references()
    
    if not _deadline_ok():
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Multi-start: try several initial points
    # Start 1: constant gamma ~ 0.55
    # Start 2: based on typical gamma(a) behavior
    # Start 3-5: random perturbations
    
    initial_points = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.50, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.60, -0.1, 0.05, 0.0, 0.0, 0.0],
        [0.55, 0.05, -0.02, 0.02, 0.0, 0.0],
        [0.45, 0.2, 0.0, -0.1, 0.0, 0.05],
    ]
    
    # Add some random starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        init = 0.55 + rng.uniform(-0.3, 0.3, size=6)
        init = np.clip(init, -10, 10)
        initial_points.append(init)
    
    best_coeffs = initial_points[0]
    best_err = _objective_from_coeffs(best_coeffs)
    
    for init in initial_points:
        if not _deadline_ok():
            break
        
        c_opt, err_opt = _coordinate_descent(init, max_sweeps=150)
        
        if err_opt < best_err:
            best_err = err_opt
            best_coeffs = c_opt
    
    # Final refinement with tighter coordinate descent
    if _deadline_ok():
        c_opt, err_opt = _coordinate_descent(best_coeffs, max_sweeps=300)
        if err_opt < best_err:
            best_err = err_opt
            best_coeffs = c_opt
    
    best_coeffs = np.clip(best_coeffs, -10, 10)
    return list(best_coeffs)
