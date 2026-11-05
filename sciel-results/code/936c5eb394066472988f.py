"""Growth-index approximation for CPL dark-energy cosmologies.

Structural change: Use a log-space linear LSQ pre-fit to exploit the
linearity of ln(f) = gamma * ln(Om_m) in the coefficients, providing a
much better initial point for the subsequent minimax polish. The key
insight is that the growth index gamma has a well-known analytic
structure in CPL models that can be captured more efficiently by fitting
in a transformed space where the error landscape is smoother.

Strategy:
1. Pre-compute reference ODE solutions on a dense Chebyshev grid.
2. Build linear LSQ system for gamma from ln(f) residuals.
3. Solve for 6 coefficients via least squares.
4. Polish with Nelder-Mead on exact minimax objective.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

_START_TIME = time.time()
_TIME_LIMIT = 85.0

def _deadline_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

def _chebyshev_nodes(n):
    k = np.arange(n)
    return np.cos(np.pi * (2 * k + 1) / (2 * n))

# ── Physics ─────────────────────────────────────────────────────────────────

def _omega_m(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    a = np.clip(a, 1e-12, 1e6)
    a3 = a ** 3
    de_ratio = a3 ** (-(1.0 + w0 + wa)) * np.exp(-3.0 * wa * (a - 1.0))
    om_m = Om0 / (Om0 + (1.0 - Om0) * de_ratio)
    return np.clip(om_m, 1e-15, 1.0)

def _gamma(c, a, Om0, w0, wa):
    Om_m = _omega_m(a, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    c0, c1, c2, c3, c4, c5 = c
    return c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q

def _f_model(c, a, Om0, w0, wa):
    Om_m = _omega_m(a, Om0, w0, wa)
    gam = _gamma(c, a, Om0, w0, wa)
    return Om_m ** gam

def _D_model(c, a, Om0, w0, wa):
    x = np.log(a)
    f_vals = _f_model(c, a, Om0, w0, wa)
    dx = np.diff(x)
    f_avg = 0.5 * (f_vals[1:] + f_vals[:-1])
    cum = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
    J = cum[-1] - cum
    return np.exp(-J)

# ── Reference ODE solution ──────────────────────────────────────────────────

def _reference(a_grid, Om0, w0, wa):
    x_min = np.log(a_grid[0])
    x_max = np.log(a_grid[-1])
    
    def ode(x, y):
        a_val = np.exp(x)
        Om_m = _omega_m(a_val, Om0, w0, wa)
        w = w0 + wa * (1.0 - a_val)
        dOm_dx = -3.0 * Om_m * (Om_m - 1.0)
        D, F = y
        Dpp = -(1.5 + (1.0 + w) - 1.5 * dOm_dx) * F + 1.5 * Om_m * D
        return [F, Dpp]
    
    a0 = a_grid[0]
    D0 = a0 ** (5.0 / 6.0)
    F0 = (5.0 / 6.0) * a0 ** (-1.0 / 6.0)
    
    try:
        sol = solve_ivp(ode, [x_min, x_max], [D0, F0],
                        t_eval=a_grid, max_step=0.001, rtol=1e-10, atol=1e-12)
        if not sol.success:
            raise RuntimeError("ODE failed")
        D = sol.y[0]
        F = sol.y[1]
        D = D / D[-1]
        return F, D
    except Exception:
        gam_ref = 0.55
        Om_m = _omega_m(a_grid, Om0, w0, wa)
        f_ref = Om_m ** gam_ref
        x = np.log(a_grid)
        dx = np.diff(x)
        f_avg = 0.5 * (f_ref[1:] + f_ref[:-1])
        cum = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
        J = cum[-1] - cum
        D = np.exp(-J)
        D = D / D[-1]
        return f_ref, D

# ── Validation cosmologies ──────────────────────────────────────────────────

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
]

_N_POINTS = 512
_cheb_x = _chebyshev_nodes(_N_POINTS)
_log_a_min = np.log(1e-4)
_log_a_max = 0.0
_EVAL_A = np.exp(_log_a_min + (_cheb_x + 1) / 2 * (_log_a_max - _log_a_min))
_EVAL_A = np.sort(_EVAL_A)

# Pre-compute reference solutions
_REF = []
for (Om0, w0, wa) in _VALIDATION:
    f_r, D_r = _reference(_EVAL_A, Om0, w0, wa)
    _REF.append((f_r, D_r, Om0, w0, wa))

# ── Objective functions ─────────────────────────────────────────────────────

def _objective(c):
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
        
        f_m = _f_model(c, _EVAL_A, Om0, w0, wa)
        D_m = _D_model(c, _EVAL_A, Om0, w0, wa)
        
        rel_f = np.abs(f_m - f_r) / np.maximum(np.abs(f_r), 1e-12)
        rel_D = np.abs(D_m - D_r) / np.maximum(np.abs(D_r), 1e-12)
        
        max_err = max(max_err, np.max(rel_f), np.max(rel_D))
    
    return 100.0 * max_err

def _smooth_objective(c, temperature=0.003):
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6
    
    all_errors = []
    for f_r, D_r, Om0, w0, wa in _REF:
        gam = _gamma(c, _EVAL_A, Om0, w0, wa)
        if np.any(gam < 0.1) or np.any(gam > 1.2):
            viol = np.maximum(0.0, 0.1 - gam) + np.maximum(0.0, gam - 1.2)
            all_errors.append(5.0 * viol)
            continue
        
        f_m = _f_model(c, _EVAL_A, Om0, w0, wa)
        D_m = _D_model(c, _EVAL_A, Om0, w0, wa)
        
        rel_f = np.abs(f_m - f_r) / np.maximum(np.abs(f_r), 1e-12)
        rel_D = np.abs(D_m - D_r) / np.maximum(np.abs(D_r), 1e-12)
        
        all_errors.append(rel_f)
        all_errors.append(rel_D)
    
    all_errors = np.concatenate(all_errors)
    if len(all_errors) == 0:
        return 1e6
    
    scaled = all_errors / temperature
    max_s = np.max(scaled)
    if max_s > 700:
        log_sum_exp = max_s
    else:
        log_sum_exp = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    
    return 100.0 * temperature * log_sum_exp

# ── Log-space LSQ pre-fit ───────────────────────────────────────────────────

def _lsq_prefit():
    """Exploit ln(f) = gamma * ln(Om_m) linearity for a better initial guess."""
    # Build design matrix for gamma: gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # For each (a, Om0, w0, wa): ln(f_ref) = gamma * ln(Om_m)
    # So gamma_ref = ln(f_ref) / ln(Om_m)
    # We want to fit gamma(c, a, Om0, w0, wa) ≈ gamma_ref
    
    A = []
    b = []
    for f_r, D_r, Om0, w0, wa in _REF:
        Om_m = _omega_m(_EVAL_A, Om0, w0, wa)
        q = 1.0 - Om_m
        u = 1.0 + w0 + wa * (1.0 - _EVAL_A)
        
        # gamma_ref = ln(f_r) / ln(Om_m)
        with np.errstate(divide='ignore', invalid='ignore'):
            gamma_ref = np.where(np.abs(np.log(Om_m)) > 1e-10, 
                                np.log(f_r) / np.log(Om_m), 
                                0.55)
        
        for i in range(len(_EVAL_A)):
            row = np.array([1.0, q[i], q[i]**2, u[i], wa * _EVAL_A[i], u[i] * q[i]])
            A.append(row)
            b.append(gamma_ref[i])
    
    A = np.array(A)
    b = np.array(b)
    
    # Weighted LSQ: weight by 1/Om_m to emphasize low-a region where errors matter
    weights = 1.0 / _omega_m(_EVAL_A, 0.3, -1.0, 0.0)  # representative
    w = np.repeat(weights, len(_VALIDATION))
    
    try:
        c_lsq, residuals, rank, sv = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
        return np.clip(c_lsq, -10, 10)
    except Exception:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

# ── Optimization ────────────────────────────────────────────────────────────

def _optimize():
    # Stage 1: Log-space LSQ pre-fit
    if _deadline_ok():
        c_lsq = _lsq_prefit()
        best_c = c_lsq.copy()
        best_err = _objective(best_c)
    else:
        best_c = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
        best_err = _objective(best_c)
    
    # Stage 2: Nelder-Mead on smooth objective from LSQ start
    if _deadline_ok():
        try:
            res = minimize(_smooth_objective, best_c, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True})
            fun = _objective(res.x)
            if fun < best_err:
                best_err = fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Stage 3: L-BFGS-B from multiple starts
    rng = np.random.default_rng(42)
    start_points = [best_c]
    for i in range(8):
        if not _deadline_ok():
            break
        scale = 0.05 * (1.0 - i / 10.0)
        pert = rng.normal(0, scale, 6)
        x0 = np.clip(best_c + pert, -5, 5)
        start_points.append(x0)
    
    for x0 in start_points[1:]:
        if not _deadline_ok():
            break
        try:
            res = minimize(_smooth_objective, x0, method='L-BFGS-B',
                           bounds=[(-5, 5)] * 6,
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            fun = _objective(res.x)
            if fun < best_err:
                best_err = fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Stage 4: Exact minimax with SLSQP
    if _deadline_ok():
        try:
            res = minimize(_objective, best_c, method='SLSQP',
                           bounds=[(-5, 5)] * 6,
                           options={'maxiter': 500, 'ftol': 1e-12})
            fun = _objective(res.x)
            if fun < best_err:
                best_err = fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Stage 5: Final Nelder-Mead polish on exact objective
    if _deadline_ok():
        try:
            res = minimize(_objective, best_c, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_err:
                best_err = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    return np.clip(best_c, -10, 10)

_COEFFS = _optimize()

def construct():
    return list(_COEFFS)
