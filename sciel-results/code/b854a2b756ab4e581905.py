"""Growth-index approximation for CPL dark-energy cosmologies.

Structural change: reference-cosmology anchoring.
Instead of directly optimizing the six polynomial coefficients, we:
1. Solve the growth ODE for a diverse set of anchor cosmologies to get
   high-accuracy gamma_ref(a) profiles.
2. For each anchor, project its gamma profile onto the six-dimensional
   ansatz basis to get a set of "anchor coefficient vectors".
3. The final six coefficients are a convex combination (weights sum to 1,
   non-negative) of these anchor vectors.
4. Optimize the weights (a small convex-like problem) to minimize the
   worst-case relative error over validation cosmologies.

This transforms the search into interpolation within a physically
motivated subspace, which should be more robust and converge faster.
"""

import numpy as np
from scipy.optimize import minimize
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
        D, F = y
        D_double = -(1.5 + (1.0 + w) - 1.5 * dOm_m_dx) * F + 1.5 * Om_m * D
        return [F, D_double]

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

# ── Anchor cosmologies for projection ────────────────────────────────────────

_ANCHOR_COSMOLOGIES = [
    (0.30, -1.0, 0.0),
    (0.30, -0.5, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.25, -1.0, 0.5),
    (0.35, -1.0, -0.3),
    (0.30, -1.2, 0.4),
    (0.30, -0.9, -0.4),
    (0.30, -0.7, 0.2),
    (0.28, -1.0, 1.0),
]

# ── Evaluation grid ──────────────────────────────────────────────────────────

_EVAL_A = np.logspace(np.log10(1e-3), 0.0, 200)

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

# ── Pre-compute reference data ───────────────────────────────────────────────

_REF_DATA = []
_ANCHOR_COEFFS = []  # Each is a 6-dim vector of coefficients that best fit that anchor's gamma

def _build_all():
    global _REF_DATA, _ANCHOR_COEFFS
    _REF_DATA = []
    _ANCHOR_COEFFS = []
    
    # Build reference data for validation cosmologies
    for (Om0, w0, wa) in _VALIDATION_COSMOLOGIES:
        try:
            f_ref, D_ref = _growth_ode_reference(_EVAL_A, Om0, w0, wa)
            _REF_DATA.append((f_ref, D_ref, Om0, w0, wa))
        except Exception:
            _REF_DATA.append(None)
    
    # Build anchor coefficient vectors by projecting gamma_ref onto the basis
    a_eval = _EVAL_A
    for (Om0, w0, wa) in _ANCHOR_COSMOLOGIES:
        try:
            # Get reference f and D
            f_ref, D_ref = _growth_ode_reference(a_eval, Om0, w0, wa)
            
            # Extract gamma_ref from f_ref: f = Om_m^gamma => gamma = ln(f)/ln(Om_m)
            Om_m_eval = _omega_m_of_a(a_eval, Om0, w0, wa)
            Om_m_safe = np.clip(Om_m_eval, 1e-15, 1.0)
            f_safe = np.clip(f_ref, 1e-15, 1.0)
            ln_Om = np.log(Om_m_safe)
            ln_f = np.log(f_safe)
            
            # gamma_ref = ln_f / ln_Om (where ln_Om != 0)
            valid = np.abs(ln_Om) > 1e-6
            gamma_ref = np.zeros_like(a_eval)
            gamma_ref[valid] = ln_f[valid] / ln_Om[valid]
            # Fill invalid points with interpolation
            if np.sum(valid) > 2:
                from scipy.interpolate import interp1d
                interp_func = interp1d(a_eval[valid], gamma_ref[valid], fill_value='extrapolate')
                gamma_ref[~valid] = interp_func(a_eval[~valid])
            else:
                gamma_ref[~valid] = 0.55
            
            # Project gamma_ref onto the 6-dim basis
            # Basis functions: [1, q, q^2, u, wa*a, u*q]
            q = 1.0 - Om_m_eval
            u = 1.0 + w0 + wa * (1.0 - a_eval)
            basis = np.column_stack([
                np.ones_like(a_eval),
                q,
                q**2,
                u,
                wa * a_eval,
                u * q
            ])
            
            # Least squares fit: gamma_ref ≈ basis @ coeffs
            coeffs, residuals, rank, sv = np.linalg.lstsq(basis, gamma_ref, rcond=None)
            coeffs = np.clip(coeffs, -10, 10)
            _ANCHOR_COEFFS.append(coeffs)
        except Exception:
            # Fallback: constant gamma
            _ANCHOR_COEFFS.append(np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0]))
    
    if not _deadline_ok():
        return False
    return True

# ── Objective function ───────────────────────────────────────────────────────

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
    
    for idx, ref in enumerate(_REF_DATA):
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

        f_avg = 0.5 * (f_model[1:] + f_model[:-1])
        cum_int = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
        integral_vals = cum_int - cum_int[-1]
        D_model = np.exp(integral_vals)
        D_model = D_model / D_model[-1]

        err_D = np.abs(D_model - D_ref) / np.abs(D_ref)
        max_err = max(max_err, np.max(err_D))

        if not _deadline_ok():
            return max_err

    return 100.0 * max_err

def _objective_from_weights(weights):
    """Objective in terms of anchor weights (convex combination)."""
    weights = np.asarray(weights, dtype=float)
    if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
        return 1e6
    
    # Normalize weights to be non-negative and sum to 1
    w = np.clip(weights, 0, None)
    w_sum = np.sum(w)
    if w_sum < 1e-10:
        w = np.ones(len(w)) / len(w)
    else:
        w = w / w_sum
    
    # Compute coefficients as weighted average of anchor coefficients
    c = np.zeros(6)
    for i, coeff in enumerate(_ANCHOR_COEFFS):
        c += w[i] * coeff
    
    return _objective_from_coeffs(c)

# ── Construct ────────────────────────────────────────────────────────────────

def construct():
    if not _deadline_ok():
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    if not _build_all():
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    n_anchors = len(_ANCHOR_COEFFS)
    if n_anchors == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Initial: uniform weights
    best_weights = np.ones(n_anchors) / n_anchors
    best_coeffs = np.zeros(6)
    for i, coeff in enumerate(_ANCHOR_COEFFS):
        best_coeffs += best_weights[i] * coeff
    best_err = _objective_from_coeffs(best_coeffs)
    
    # Optimize weights using Nelder-Mead on the weight space
    def weight_obj(w):
        if not _deadline_ok():
            return best_err
        return _objective_from_weights(w)
    
    try:
        result = minimize(weight_obj, best_weights, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
        if result.fun < best_err:
            best_weights = np.clip(result.x, 0, None)
            ws = np.sum(best_weights)
            if ws > 1e-10:
                best_weights = best_weights / ws
            best_coeffs = np.zeros(6)
            for i, coeff in enumerate(_ANCHOR_COEFFS):
                best_coeffs += best_weights[i] * coeff
            best_err = result.fun
    except Exception:
        pass
    
    # Second pass: direct coefficient optimization from the best interpolated point
    if _deadline_ok():
        def coeff_obj(c):
            if not _deadline_ok():
                return best_err
            return _objective_from_coeffs(c)
        
        try:
            result = minimize(coeff_obj, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if result.fun < best_err:
                best_coeffs = result.x
                best_err = result.fun
        except Exception:
            pass
    
    # Final refinement
    if _deadline_ok():
        def coeff_obj2(c):
            if not _deadline_ok():
                return best_err
            return _objective_from_coeffs(c)
        
        try:
            result = minimize(coeff_obj2, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            if result.fun < best_err:
                best_coeffs = result.x
                best_err = result.fun
        except Exception:
            pass
    
    best_coeffs = np.clip(best_coeffs, -10, 10)
    return list(best_coeffs)
