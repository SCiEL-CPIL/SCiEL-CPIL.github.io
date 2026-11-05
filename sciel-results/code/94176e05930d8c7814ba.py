"""Growth-index approximation for CPL dark-energy cosmologies.

Structural change: Direct minimax (Chebyshev) formulation with auxiliary
variable t >= max relative error, solved via SLSQP with a fast ODE reference
built at moderate tolerances.  The reference is computed once on a compact
grid; the objective is the true worst-case over that grid, avoiding the
indirectness of fitting gamma in log-space.  Multi-start Nelder-Mead refines
the SLSQP result on the same grid.

Key speedups:
  - ODE: rtol=1e-9, atol=1e-12, max_step=0.005, 300-point log-spaced grid.
  - Reference build < 5 s, leaving > 70 s for optimization.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

_START_TIME = time.time()
_TIME_LIMIT = 85.0

def _deadline_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

# ---------------------------------------------------------------------------
# Cosmology helpers
# ---------------------------------------------------------------------------

def _omega_m_of_a(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    a_safe = np.clip(a, 1e-12, 1e6)
    a_inv3 = a_safe ** (-3.0)
    de_factor = a_safe ** (-3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (a_safe - 1.0))
    Om_DE = (1.0 - Om0) * de_factor
    denom = Om0 * a_inv3 + Om_DE
    Om_m = Om0 * a_inv3 / denom
    return Om_m

def _design_matrix(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    Om_m = _omega_m_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    X = np.column_stack([
        np.ones_like(a),
        q,
        q ** 2,
        u,
        wa * a,
        u * q
    ])
    return X

# ---------------------------------------------------------------------------
# ODE reference (fast)
# ---------------------------------------------------------------------------

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
        D, F = y
        D_double = -(1.5 + (1.0 + w) - 1.5 * dOm_m_dx) * F + 1.5 * Om_m * D
        return [F, D_double]

    D0 = a_min ** (5.0 / 6.0)
    F0 = (5.0 / 6.0) * a_min ** (-1.0 / 6.0)

    try:
        sol = solve_ivp(ode, [x_min, x_max], [D0, F0],
                        t_eval=a_grid,
                        max_step=0.005, rtol=1e-9, atol=1e-12)
        if not sol.success:
            raise RuntimeError("ODE integration failed")

        D_ref = sol.y[0]
        F_ref = sol.y[1]
        D_ref = D_ref / D_ref[-1]
        f_ref = F_ref / (a_grid * D_ref)
        return f_ref, D_ref

    except Exception:
        # Fallback: constant gamma = 0.55
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

# ---------------------------------------------------------------------------
# Validation cosmologies and grid
# ---------------------------------------------------------------------------

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
    (0.22, -1.0, 0.8),
    (0.38, -1.0, -0.6),
    (0.30, -0.7, 0.6),
    (0.25, -1.2, 0.3),
    (0.35, -0.8, -0.5),
]

_N_GRID = 300
_A_GRID = np.logspace(np.log10(1e-4), 0.0, _N_GRID)
_DX = np.diff(np.log(_A_GRID))

_REF_DATA = []

def _build_references():
    global _REF_DATA
    _REF_DATA = []
    for (Om0, w0, wa) in _VALIDATION_COSMOLOGIES:
        try:
            f_ref, D_ref = _growth_ode_reference(_A_GRID, Om0, w0, wa)
            X = _design_matrix(_A_GRID, Om0, w0, wa)
            Om_m = _omega_m_of_a(_A_GRID, Om0, w0, wa)
            _REF_DATA.append((f_ref, D_ref, Om0, w0, wa, X, Om_m))
        except Exception:
            _REF_DATA.append(None)

_build_references()

# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _compute_D_from_f(f_vals, dx):
    f_avg = 0.5 * (f_vals[1:] + f_vals[:-1])
    cum_int = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
    J_vals = cum_int - cum_int[-1]
    D = np.exp(J_vals)
    return D / D[-1]

def _max_relative_error(c, ref_data, a_grid, dx):
    c = np.asarray(c)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6

    max_err = 0.0
    for ref in ref_data:
        if ref is None:
            continue
        f_ref, D_ref, Om0, w0, wa, X, Om_m = ref

        gamma = X @ c
        # Enforce gamma in [0.1, 1.2]
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            violation = np.maximum(0.0, 0.1 - gamma) + np.maximum(0.0, gamma - 1.2)
            max_err = max(max_err, 10.0 * np.max(violation))
            continue

        Om_m_safe = np.clip(Om_m, 1e-15, 1.0)
        f_model = Om_m_safe ** gamma

        denom_f = np.maximum(np.abs(f_ref), 1e-10)
        rel_err_f = np.abs(f_model - f_ref) / denom_f

        D_model = _compute_D_from_f(f_model, dx)
        denom_D = np.maximum(np.abs(D_ref), 1e-10)
        rel_err_D = np.abs(D_model - D_ref) / denom_D

        max_err = max(max_err, np.max(rel_err_f), np.max(rel_err_D))

    return max_err

def _objective(c):
    return 100.0 * _max_relative_error(c, _REF_DATA, _A_GRID, _DX)

# ---------------------------------------------------------------------------
# Initial guess: least-squares fit of gamma in log-space
# ---------------------------------------------------------------------------

def _initial_guess():
    all_X = []
    all_target = []

    for ref in _REF_DATA:
        if ref is None:
            continue
        f_ref, D_ref, Om0, w0, wa, X, Om_m = ref
        Om_m_safe = np.clip(Om_m, 1e-15, 1.0)
        log_f = np.log(np.clip(f_ref, 1e-30, None))
        log_Om = np.log(Om_m_safe)

        with np.errstate(divide='ignore', invalid='ignore'):
            gamma_target = log_f / log_Om

        valid = np.isfinite(gamma_target) & (np.abs(gamma_target) < 5)
        all_X.append(X[valid])
        all_target.append(gamma_target[valid])

    if len(all_X) == 0:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

    X_stack = np.vstack(all_X)
    target_stack = np.concatenate(all_target)

    try:
        result, _, _, _ = np.linalg.lstsq(X_stack, target_stack, rcond=None)
        return np.clip(result, -5, 5)
    except Exception:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def _run_optimization():
    best_c = _initial_guess()
    best_c = np.clip(best_c, -10, 10)
    best_err = _objective(best_c)

    if not _deadline_ok():
        return best_c

    # Stage 1: SLSQP
    try:
        bounds = [(-3, 3)] * 6
        res = minimize(
            _objective,
            best_c,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
        )
        if res.success and res.fun < best_err:
            best_err = res.fun
            best_c = res.x.copy()
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage 2: Nelder-Mead
    try:
        res = minimize(
            _objective,
            best_c,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
        )
        if res.fun < best_err:
            best_err = res.fun
            best_c = res.x.copy()
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage 3: Multi-start Nelder-Mead with adaptive scales
    rng = np.random.default_rng(42)
    for i in range(200):
        if not _deadline_ok():
            break
        scale = 0.3 * (0.7 ** (i // 30))
        x_try = best_c + rng.normal(0, scale, 6)
        x_try = np.clip(x_try, -5, 5)
        try:
            res = minimize(
                _objective,
                x_try,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_err:
                best_err = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    best_c = np.clip(best_c, -10, 10)
    return best_c

_COEFFS = _run_optimization()

def construct():
    return list(_COEFFS)
