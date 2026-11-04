"""Model-and-optimize: parameterize 12 double roots, compute C4 bound via polynomial moments, optimize.
Uses log-variable change to remove positivity constraint, efficient gradient via Q structure,
alpha-based init + multi-start L-BFGS-B with tight polish."""
from scipy.optimize import differential_evolution
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize
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

def _build_q(roots):
    """Build Q(x) = prod (x-x_i) and return Q_coeffs."""
    from numpy.polynomial.polynomial import polyfromroots
    return polyfromroots(roots)

def _build_p_and_p2(roots):
    """Build P = Q^2 and P2 = Q^4, return (P, P2)."""
    from numpy.polynomial.polynomial import polymul
    Q = _build_q(roots)
    P = polymul(Q, Q)
    P2 = polymul(P, P)
    return P, P2

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx.
    Uses the structure P = Q^2 for efficient gradient computation."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    Q = _build_q(roots)
    P = np.convolve(Q, Q)  # P = Q^2
    P2 = np.convolve(P, P)  # P2 = P^2 = Q^4
    
    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)
    
    # Efficient gradient using logarithmic derivative approach:
    # For Q = prod_j (x - x_j), we have Q'(x)/Q(x) = sum_i 1/(x - x_i)
    # dQ/dx_i = Q * prod_{j!=i} (x - x_j) / prod_{j!=i} (x_j - x_i) * ... 
    # Actually simpler: dQ/dx_i = Q / (x - x_i)
    # So dP/dx_i = 2*P/(x-x_i), d(P^2)/dx_i = 4*P^2/(x-x_i)
    #
    # We need: int [P/(x-x_i)] e^{-x} dx and int [P^2/(x-x_i)] e^{-x} dx
    # Use partial fractions or direct polynomial division
    
    from numpy.polynomial.polynomial import polydiv
    
    dI1 = np.zeros(n)
    dI2 = np.zeros(n)
    
    for i in range(n):
        # S_i = P / (x - x_i), T_i = P^2 / (x - x_i)
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
    # Chain rule: dC4/d(log_x_i) = dC4/dx_i * x_i
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

    # Phase 1.5: Global search using Differential Evolution in log-space
    # This helps escape local minima that L-BFGS-B might get stuck in
    if time_ok():
        try:
            log_bounds = [(math.log(1e-4), math.log(1e4))] * 12
            # Run DE with a few different seeds to explore different basins
            de_seeds = [42, 123, 999]
            for de_seed in de_seeds:
                if not time_ok():
                    break
                de_res = differential_evolution(
                    _objective_log_safe,
                    log_bounds,
                    seed=de_seed,
                    maxiter=50,
                    popsize=10,
                    tol=1e-6,
                    polish=False
                )
                if np.all(np.isfinite(de_res.x)):
                    candidate = np.exp(de_res.x)
                    candidate = np.sort(candidate)
                    c4 = _compute_c4(candidate)
                    if np.isfinite(c4) and c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 2: L-BFGS-B refinement in log-space (unconstrained)
    if time_ok():
        try:
            log_init = np.log(best_roots.copy())
            res = minimize(
                _objective_log_safe,
                log_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 100}
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

    # Phase 3: Multi-start L-BFGS-B in log-space from perturbed best
    if time_ok():
        candidates_to_refine = [np.log(best_roots.copy())]
        rng = np.random.default_rng(42)
        
        # Perturbations in log-space with larger scales to escape basin
        for scale in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0]:
            for _ in range(5):
                pert = np.log(best_roots) + scale * rng.standard_normal(12)
                candidates_to_refine.append(pert)
        
        # Also try different scaling in original space
        for factor in [0.2, 0.4, 0.6, 1.5, 2.0, 3.0, 5.0]:
            pert_roots = best_roots * factor
            pert_roots = np.maximum(pert_roots, 1e-3)
            candidates_to_refine.append(np.log(pert_roots))
        
        # Try completely different initializations from various alpha values
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            try:
                xg, _ = roots_genlaguerre(12, alpha)
                xg = np.sort(xg)
                if np.all(xg > 0) and np.isfinite(xg).all():
                    candidates_to_refine.append(np.log(xg))
            except Exception:
                pass
        
        # Add results from DE if they are distinct from current best
        # (DE already updated best_roots if it found something better, so best_roots is the best of all)
        # We don't store DE intermediate results, but we can re-run DE with polish=True?
        # No, DE with polish=True is slower.
        # Let's just rely on the fact that best_roots is now the best of (Laguerre, DE, Polish).
        
        for init in candidates_to_refine:
            if not time_ok():
                break
            try:
                res = minimize(
                    _objective_log_safe,
                    init,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 200}
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
    for polish_iter in range(15):
        if not time_ok():
            break
        try:
            log_best = np.log(best_roots)
            res = minimize(
                _objective_log_safe,
                log_best,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-17, 'maxls': 300}
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
        for polish_iter in range(8):
            if not time_ok():
                break
            try:
                log_best = np.log(best_roots)
                res = minimize(
                    _objective_log_safe,
                    log_best,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 20000, 'ftol': 1e-20, 'gtol': 1e-19, 'maxls': 500}
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
# EVOLVE-BLOCK-END
