"""Model-and-optimize: parameterize 12 double roots, compute C4 bound via polynomial moments, optimize.
Uses 1D search over generalized Laguerre alpha + DE global search + high-precision L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_laguerre, roots_genlaguerre

# Precompute factorials for moment computation
_FACTS = np.array([math.factorial(k) for k in range(60)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (given in ascending coefficient order) * e^{-x} dx over [0,inf)."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    m = min(n, 60)
    return float(np.dot(coeffs[:m], _FACTS[:m]))

def _poly_moment_fast(coeffs):
    """Faster moment using Horner-like accumulation to avoid array allocation."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    m = min(n, 60)
    # Direct dot product is already fast; this is equivalent
    return float(np.dot(coeffs[:m], _FACTS[:m]))

def _build_p_squared_coeffs(roots):
    """Given 12 roots, build P(x) = prod (x-x_i)^2 and return (P_coeffs, P2_coeffs)."""
    from numpy.polynomial.polynomial import polyfromroots, polymul
    B = polyfromroots(roots)
    P = polymul(B, B)
    P2 = polymul(P, P)
    return P, P2

def _build_q_and_p(roots):
    """Build Q = prod(x-x_i) and P = Q^2, return (Q, P, P2)."""
    from numpy.polynomial.polynomial import polyfromroots, polymul
    Q = polyfromroots(roots)
    P = polymul(Q, Q)
    P2 = polymul(P, P)
    return Q, P, P2

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    P, P2 = _build_p_squared_coeffs(roots)
    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)
    
    # More efficient gradient: use Q = prod(x-x_i), P = Q^2
    # dP/dx_i = 2*Q * dQ/dx_i = 2*Q * prod_{j!=i}(x-x_j) = 2*Q^2/(x-x_i) = 2*P/(x-x_i)
    # Wait: dQ/dx_i = prod_{j!=i}(x-x_j) = Q/(x-x_i)
    # So dP/dx_i = 2*Q * Q/(x-x_i) = 2*P/(x-x_i)
    # 
    # I1 = int P e^{-x} dx
    # dI1/dx_i = int [2*P/(x-x_i)] e^{-x} dx
    # I2 = int P^2 e^{-x} dx
    # dI2/dx_i = 2*int P * [2*P/(x-x_i)] e^{-x} dx = 4*int [P^2/(x-x_i)] e^{-x} dx
    #
    # dC4/dx_i = dI2/I1^2 - 2*I2*dI1/I1^3

    from numpy.polynomial.polynomial import polydiv
    
    dI1 = np.zeros(n)
    dI2 = np.zeros(n)
    
    for i in range(n):
        divisor = np.array([-roots[i], 1.0])
        S_i, rem = polydiv(P, divisor)
        T_i, rem2 = polydiv(P2, divisor)
        
        int_S = _poly_moment(S_i)
        int_T = _poly_moment(T_i)
        
        dI1[i] = 2.0 * int_S
        dI2[i] = 4.0 * int_T
    
    I1_sq = I1 * I1
    I1_cu = I1_sq * I1
    grad = dI2 / I1_sq - 2.0 * I2 * dI1 / I1_cu
    
    return float(c4), grad

def _objective(roots):
    return _compute_c4(roots)

def _objective_grad(roots):
    c4, g = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(roots)
    return c4, g

def _objective_grad_safe(roots):
    """Wrapper that handles edge cases for scipy minimize."""
    try:
        c4, g = _compute_c4(roots, need_grad=True)
        if not np.isfinite(c4) or not np.all(np.isfinite(g)):
            return np.inf, np.zeros_like(roots)
        return c4, g
    except Exception:
        return np.inf, np.zeros_like(roots)

def _log_objective_grad(log_roots):
    """Objective in log-space: roots = exp(log_roots). Removes positivity constraint."""
    try:
        roots = np.exp(log_roots)
        c4, g = _compute_c4(roots, need_grad=True)
        if not np.isfinite(c4):
            return np.inf, np.zeros_like(log_roots)
        # Chain rule: dC4/d(log_x_i) = dC4/dx_i * x_i
        grad_log = g * roots
        if not np.all(np.isfinite(grad_log)):
            return np.inf, np.zeros_like(log_roots)
        return c4, grad_log
    except Exception:
        return np.inf, np.zeros_like(log_roots)

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 15.0

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    best_roots = None
    best_c4 = np.inf
    best_alpha = 1.0

    # Phase 1: 1D search over alpha for generalized Laguerre nodes
    alpha_values = np.concatenate([
        np.logspace(-2, 3, 50),
        np.linspace(0.1, 50, 100),
    ])
    alpha_values = np.unique(np.sort(alpha_values))
    
    for alpha in alpha_values:
        if not time_ok():
            break
        try:
            xg, _ = roots_genlaguerre(12, float(alpha))
            xg = np.sort(xg)
            if np.all(xg > 0) and np.isfinite(xg).all():
                c4g = _compute_c4(xg)
                if np.isfinite(c4g) and c4g < best_c4:
                    best_c4 = c4g
                    best_roots = xg.copy()
                    best_alpha = float(alpha)
        except Exception:
            pass
    
    # Refine alpha around best
    if best_roots is not None and time_ok():
        a_lo = max(best_alpha * 0.5, 0.001)
        a_hi = best_alpha * 2.0
        fine_alphas = np.linspace(a_lo, a_hi, 50)
        for alpha in fine_alphas:
            try:
                xg, _ = roots_genlaguerre(12, float(alpha))
                xg = np.sort(xg)
                if np.all(xg > 0) and np.isfinite(xg).all():
                    c4g = _compute_c4(xg)
                    if np.isfinite(c4g) and c4g < best_c4:
                        best_c4 = c4g
                        best_roots = xg.copy()
                        best_alpha = float(alpha)
            except Exception:
                pass
        a_lo = max(best_alpha * 0.8, 0.001)
        a_hi = best_alpha * 1.2
        fine_alphas2 = np.linspace(a_lo, a_hi, 30)
        for alpha in fine_alphas2:
            try:
                xg, _ = roots_genlaguerre(12, float(alpha))
                xg = np.sort(xg)
                if np.all(xg > 0) and np.isfinite(xg).all():
                    c4g = _compute_c4(xg)
                    if np.isfinite(c4g) and c4g < best_c4:
                        best_c4 = c4g
                        best_roots = xg.copy()
                        best_alpha = float(alpha)
            except Exception:
                pass

    # Fallback if alpha search failed
    if best_roots is None:
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
        best_c4 = _compute_c4(best_roots)

    # Phase 2: L-BFGS-B refinement in log-space from the best alpha point
    if time_ok():
        try:
            log_init = np.log(best_roots.copy())
            res = minimize(
                _log_objective_grad,
                log_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50}
            )
            if np.isfinite(res.fun):
                candidate = np.sort(np.exp(res.x))
                if np.all(candidate > 0):
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: Differential Evolution global search (limited iterations)
    if time_ok():
        try:
            # Use a moderate number of DE iterations to explore the landscape
            de_bounds = [(1e-3, 100.0)] * 12
            de_res = differential_evolution(
                _objective,
                bounds=de_bounds,
                seed=42,
                maxiter=30,
                popsize=10,
                tol=1e-10,
                mutation=(0.5, 1.5),
                recombination=0.7,
                polish=False,
                init='latinhypercube'
            )
            if np.all(de_res.x > 0) and np.isfinite(de_res.fun):
                candidate = np.sort(de_res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 4: Multi-start L-BFGS-B in log-space from perturbed best
    if time_ok():
        log_best = np.log(best_roots.copy())
        candidates_to_refine = [log_best.copy()]
        rng = np.random.default_rng(42)
        
        # Perturbations in log-space around best
        for scale in [0.02, 0.05, 0.1, 0.2, 0.5]:
            for _ in range(3):
                pert = log_best + scale * rng.standard_normal(12)
                candidates_to_refine.append(pert.copy())
        
        # Also try shifted/scaled variants in log-space
        for factor in [0.5, 1.5, 2.0, 0.3, 3.0]:
            pert = log_best + np.log(factor)
            candidates_to_refine.append(pert.copy())
        
        # Try uniform random in log-space
        for _ in range(5):
            pert = rng.uniform(-5, 5, size=12)
            candidates_to_refine.append(pert)
        
        for init in candidates_to_refine:
            if not time_ok():
                break
            try:
                res = minimize(
                    _log_objective_grad,
                    init,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50}
                )
                if np.isfinite(res.fun):
                    candidate = np.sort(np.exp(res.x))
                    if np.all(candidate > 0):
                        c4 = _compute_c4(candidate)
                        if c4 < best_c4:
                            best_c4 = c4
                            best_roots = candidate.copy()
            except Exception:
                pass

    # Phase 5: Final high-precision L-BFGS-B polish in log-space
    for polish_iter in range(10):
        if not time_ok():
            break
        try:
            log_best = np.log(best_roots)
            res = minimize(
                _log_objective_grad,
                log_best,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100}
            )
            if np.isfinite(res.fun):
                candidate = np.sort(np.exp(res.x))
                if np.all(candidate > 0):
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
                    else:
                        break
        except Exception:
            break

    # Phase 6: Very tight final polish with even tighter tolerances
    if time_ok():
        for polish_iter in range(5):
            if not time_ok():
                break
            try:
                log_best = np.log(best_roots)
                res = minimize(
                    _log_objective_grad,
                    log_best,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 10000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 200}
                )
                if np.isfinite(res.fun):
                    candidate = np.sort(np.exp(res.x))
                    if np.all(candidate > 0):
                        c4 = _compute_c4(candidate)
                        if c4 < best_c4:
                            best_c4 = c4
                            best_roots = candidate.copy()
                        else:
                            break
            except Exception:
                break

    # Final safety check
    if not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
