"""Model-and-optimize: parameterize 12 double roots, compute C4 bound via polynomial moments, optimize.
Uses log-variable change to remove positivity constraint, efficient gradient via Q structure,
alpha-based init + multi-start L-BFGS-B with tight polish."""
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import roots_laguerre, roots_genlaguerre
from numpy.polynomial.polynomial import polyfromroots, polymul, polydiv

# Precompute factorials for moment computation (degree of Q^4 is 48, need up to 48)
_FACTS = np.array([math.factorial(k) for k in range(80)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (given in ascending coefficient order) * e^{-x} dx over [0,inf)."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    m = min(n, len(_FACTS))
    return float(np.dot(coeffs[:m], _FACTS[:m]))

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx.
    P = Q^2, Q = prod (x-x_i). Uses efficient gradient via polynomial division."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    Q = polyfromroots(roots)
    P = np.convolve(Q, Q)  # P = Q^2, degree 24
    P2 = np.convolve(P, P)  # P2 = P^2 = Q^4, degree 48
    
    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)
    
    # Efficient gradient:
    # dQ/dx_i = Q / (x - x_i)
    # dP/dx_i = 2*P/(x-x_i), d(P^2)/dx_i = 4*P^2/(x-x_i)
    # dI1/dx_i = 2 * int [P/(x-x_i)] e^{-x} dx
    # dI2/dx_i = 4 * int [P^2/(x-x_i)] e^{-x} dx
    
    dI1 = np.zeros(n)
    dI2 = np.zeros(n)
    
    for i in range(n):
        divisor = np.array([-roots[i], 1.0])
        S_i, _ = polydiv(P, divisor)
        T_i, _ = polydiv(P2, divisor)
        
        int_S = _poly_moment(S_i)
        int_T = _poly_moment(T_i)
        
        dI1[i] = 2.0 * int_S
        dI2[i] = 4.0 * int_T
    
    I1_sq = I1 * I1
    I1_cu = I1_sq * I1
    grad = dI2 / I1_sq - 2.0 * I2 * dI1 / I1_cu
    
    return float(c4), grad

def _objective_log(log_roots):
    """Objective in log-space to remove positivity constraint."""
    roots = np.exp(log_roots)
    if not np.all(np.isfinite(roots)) or np.any(roots <= 0):
        return 1e10
    c4 = _compute_c4(roots)
    if not np.isfinite(c4):
        return 1e10
    return c4

def _objective_log_grad(log_roots):
    """Objective and gradient in log-space."""
    roots = np.exp(log_roots)
    if not np.all(np.isfinite(roots)) or np.any(roots <= 0):
        return 1e10, np.zeros_like(log_roots)
    c4, grad_x = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return 1e10, np.zeros_like(log_roots)
    grad_log = grad_x * roots
    if not np.all(np.isfinite(grad_log)):
        return 1e10, np.zeros_like(log_roots)
    return c4, grad_log

def _objective_log_safe(log_roots):
    """Safe wrapper for scipy minimize in log-space."""
    try:
        c4, g = _objective_log_grad(log_roots)
        if not np.isfinite(c4) or not np.all(np.isfinite(g)):
            return 1e10, np.zeros_like(log_roots)
        return c4, g
    except Exception:
        return 1e10, np.zeros_like(log_roots)

def _c4_from_alpha(alpha):
    """Compute C4 for roots of generalized Laguerre polynomial L_12^{(alpha)}."""
    try:
        xg, _ = roots_genlaguerre(12, float(alpha))
        xg = np.sort(xg)
        if not np.all(xg > 0) or not np.isfinite(xg).all():
            return np.inf
        return _compute_c4(xg)
    except Exception:
        return np.inf

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 30.0

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    best_roots = None
    best_c4 = np.inf
    best_alpha = 1.0

    # Phase 1: 1D search over alpha for generalized Laguerre nodes
    # Coarse grid search
    alpha_values = np.concatenate([
        np.logspace(-3, 4, 80),
        np.linspace(0.01, 100, 100),
    ])
    alpha_values = np.unique(np.sort(alpha_values))
    
    for alpha in alpha_values:
        if not time_ok():
            break
        c4g = _c4_from_alpha(alpha)
        if np.isfinite(c4g) and c4g < best_c4:
            best_c4 = c4g
            xg, _ = roots_genlaguerre(12, float(alpha))
            best_roots = np.sort(xg)
            best_alpha = float(alpha)
    
    # Refine alpha with scalar minimization around best
    if best_roots is not None and time_ok():
        try:
            a_lo = max(best_alpha * 0.3, 0.001)
            a_hi = best_alpha * 3.0
            res_alpha = minimize_scalar(
                lambda la: _c4_from_alpha(np.exp(la)),
                bounds=(math.log(a_lo), math.log(a_hi)),
                method='bounded',
                options={'xatol': 1e-12, 'maxiter': 1000}
            )
            if res_alpha.success and np.isfinite(res_alpha.fun) and res_alpha.fun < best_c4:
                best_alpha = float(np.exp(res_alpha.x))
                xg, _ = roots_genlaguerre(12, best_alpha)
                best_roots = np.sort(xg)
                best_c4 = float(res_alpha.fun)
        except Exception:
            pass
        
        # Even finer refinement
        if time_ok():
            try:
                a_lo = max(best_alpha * 0.9, 0.001)
                a_hi = best_alpha * 1.1
                res_alpha2 = minimize_scalar(
                    lambda la: _c4_from_alpha(np.exp(la)),
                    bounds=(math.log(a_lo), math.log(a_hi)),
                    method='bounded',
                    options={'xatol': 1e-14, 'maxiter': 2000}
                )
                if res_alpha2.success and np.isfinite(res_alpha2.fun) and res_alpha2.fun < best_c4:
                    best_alpha = float(np.exp(res_alpha2.x))
                    xg, _ = roots_genlaguerre(12, best_alpha)
                    best_roots = np.sort(xg)
                    best_c4 = float(res_alpha2.fun)
            except Exception:
                pass

    # Fallback if alpha search failed
    if best_roots is None:
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
        best_c4 = _compute_c4(best_roots)

    # Also try standard Laguerre (alpha=0)
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    c4_0 = _compute_c4(x0)
    if c4_0 < best_c4:
        best_c4 = c4_0
        best_roots = x0.copy()

    # Phase 2: L-BFGS-B refinement in log-space (unconstrained)
    if time_ok():
        try:
            log_init = np.log(best_roots.copy())
            res = minimize(
                _objective_log_safe,
                log_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 100}
            )
            if np.all(np.isfinite(res.x)):
                candidate = np.exp(res.x)
                candidate = np.sort(candidate)
                c4 = _compute_c4(candidate)
                if np.isfinite(c4) and c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: Multi-start L-BFGS-B in log-space from diverse initializations
    if time_ok():
        candidates_to_refine = [np.log(best_roots.copy())]
        rng = np.random.default_rng(42)
        
        # Perturbations in log-space with varying scales
        for scale in [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0]:
            for _ in range(3):
                if not time_ok():
                    break
                pert = np.log(best_roots) + scale * rng.standard_normal(12)
                candidates_to_refine.append(pert)
        
        # Different scaling in original space
        for factor in [0.2, 0.3, 0.5, 0.7, 1.5, 2.0, 3.0, 5.0, 10.0]:
            pert_roots = best_roots * factor
            pert_roots = np.maximum(pert_roots, 1e-3)
            candidates_to_refine.append(np.log(pert_roots))
        
        # Try different alpha values as initializations
        for alpha in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
            try:
                xg, _ = roots_genlaguerre(12, alpha)
                xg = np.sort(xg)
                if np.all(xg > 0) and np.isfinite(xg).all():
                    candidates_to_refine.append(np.log(xg))
            except Exception:
                pass
        
        # Random initializations
        for _ in range(10):
            if not time_ok():
                break
            random_roots = np.exp(rng.uniform(math.log(0.01), math.log(100), 12))
            candidates_to_refine.append(np.log(random_roots))
        
        for init in candidates_to_refine:
            if not time_ok():
                break
            try:
                res = minimize(
                    _objective_log_safe,
                    init,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 200}
                )
                if np.all(np.isfinite(res.x)):
                    candidate = np.exp(res.x)
                    candidate = np.sort(candidate)
                    c4 = _compute_c4(candidate)
                    if np.isfinite(c4) and c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
            except Exception:
                pass

    # Phase 4: Final high-precision L-BFGS-B polish in log-space
    for polish_iter in range(20):
        if not time_ok():
            break
        try:
            log_best = np.log(best_roots)
            res = minimize(
                _objective_log_safe,
                log_best,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 15000, 'ftol': 1e-18, 'gtol': 1e-17, 'maxls': 300}
            )
            if np.all(np.isfinite(res.x)):
                candidate = np.exp(res.x)
                candidate = np.sort(candidate)
                c4 = _compute_c4(candidate)
                if np.isfinite(c4) and c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
                else:
                    break
        except Exception:
            break

    # Phase 5: Very tight final polish with even stricter tolerances
    if time_ok():
        for polish_iter in range(10):
            if not time_ok():
                break
            try:
                log_best = np.log(best_roots)
                res = minimize(
                    _objective_log_safe,
                    log_best,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 30000, 'ftol': 1e-20, 'gtol': 1e-19, 'maxls': 500}
                )
                if np.all(np.isfinite(res.x)):
                    candidate = np.exp(res.x)
                    candidate = np.sort(candidate)
                    c4 = _compute_c4(candidate)
                    if np.isfinite(c4) and c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
                    else:
                        break
            except Exception:
                break

    # Final safety check
    if best_roots is None or not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
    
    # One more final polish if we have time
    if time_ok():
        try:
            log_best = np.log(best_roots)
            res = minimize(
                _objective_log_safe,
                log_best,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-17, 'maxls': 200}
            )
            if np.all(np.isfinite(res.x)):
                candidate = np.exp(res.x)
                candidate = np.sort(candidate)
                c4 = _compute_c4(candidate)
                if np.isfinite(c4) and c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    return [float(xi) for xi in best_roots]
