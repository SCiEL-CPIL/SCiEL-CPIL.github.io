"""Combined: model-and-optimize with efficient analytic gradient + alpha-based init + multi-start polish.
Strategy: 1) 1D search over generalized Laguerre alpha parameter, 2) multi-start DE, 3) high-precision L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution, minimize_scalar
from scipy.special import roots_laguerre, roots_genlaguerre
from numpy.polynomial.polynomial import polyfromroots, polymul

# Precompute factorials up to 80 (degree of P^2 is 48, but we need moments up to that)
_FACTS = np.array([math.factorial(k) for k in range(80)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (ascending coeff) * e^{-x} dx = sum_k coeffs[k]*k!."""
    n = min(len(coeffs), len(_FACTS))
    if n == 0:
        return 0.0
    return float(np.dot(coeffs[:n], _FACTS[:n]))

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx.
    P = prod (x-x_i)^2. Returns (c4, grad) or just c4."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    # Build B(x) = prod (x - x_i), then P = B^2, P2 = P^2 = B^4
    B = polyfromroots(roots)  # ascending order, degree 12
    P = polymul(B, B)         # degree 24
    P2 = polymul(P, P)        # degree 48

    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)

    # Analytic gradient
    # P = B^2, dP/dx_i = 2*B * dB/dx_i = -2*B*R_i where R_i = prod_{j!=i}(x-x_j)
    # dI1/dx_i = -2 * int B*R_i * e^{-x} dx
    # P2 = B^4, dP2/dx_i = 4*B^3 * dB/dx_i = -4*B^3*R_i
    # dI2/dx_i = -4 * int B^3*R_i * e^{-x} dx
    # dC4/dx_i = dI2/(I1^2) - 2*I2*dI1/(I1^3)
    
    grad = np.zeros_like(roots)
    # Precompute B^3 to avoid recomputing
    B3 = polymul(polymul(B, B), B)  # degree 36
    
    for i in range(n):
        # R_i = prod_{j!=i}(x-x_j)
        R_i = np.array([1.0])
        for j in range(n):
            if j == i:
                continue
            R_i = polymul(R_i, np.array([-roots[j], 1.0]))  # (x - x_j) in ascending: [-x_j, 1]
        
        # dI1/dx_i = -2 * int B*R_i * e^{-x} dx
        BR_i = polymul(B, R_i)  # degree 12 + 11 = 23
        dI1_i = -2.0 * _poly_moment(BR_i)
        
        # dI2/dx_i = -4 * int B^3 * R_i * e^{-x} dx
        B3R_i = polymul(B3, R_i)  # degree 36 + 11 = 47
        dI2_i = -4.0 * _poly_moment(B3R_i)
        
        grad[i] = dI2_i / (I1 * I1) - 2.0 * I2 * dI1_i / (I1 ** 3)
    
    return float(c4), grad

def _objective(roots):
    return _compute_c4(roots)

def _objective_grad(roots):
    c4, g = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(roots)
    return c4, g

def _c4_from_alpha(alpha):
    """Compute C4 for roots of generalized Laguerre polynomial L_12^{(alpha)}."""
    try:
        xg, _ = roots_genlaguerre(12, alpha)
        xg = np.sort(xg)
        if not np.all(xg > 0) or not np.isfinite(xg).all():
            return np.inf
        return _compute_c4(xg)
    except Exception:
        return np.inf

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 45.0
    
    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # Phase 1: 1D search over alpha to find optimal root density profile
    best_alpha = 0.0
    best_c4_alpha = np.inf
    
    # Grid search over log-spaced alphas
    log_alphas = np.linspace(math.log(1e-3), math.log(1e4), 300)
    for la in log_alphas:
        if not time_ok():
            break
        c4_val = _c4_from_alpha(np.exp(la))
        if c4_val < best_c4_alpha:
            best_c4_alpha = c4_val
            best_alpha = np.exp(la)

    # Refine with scalar minimization
    if time_ok():
        try:
            res_alpha = minimize_scalar(
                lambda la: _c4_from_alpha(np.exp(la)),
                bounds=(math.log(max(best_alpha * 0.5, 1e-4)), math.log(min(best_alpha * 2.0, 1e5))),
                method='bounded',
                options={'xatol': 1e-10, 'maxiter': 500}
            )
            if res_alpha.success and np.isfinite(res_alpha.fun) and res_alpha.fun < best_c4_alpha:
                best_alpha = np.exp(res_alpha.x)
                best_c4_alpha = res_alpha.fun
        except Exception:
            pass

    # Get roots from best alpha
    xg, _ = roots_genlaguerre(12, best_alpha)
    best_roots = np.sort(xg)
    best_c4 = _compute_c4(best_roots)

    # Also try standard Laguerre (alpha=0)
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    c4_0 = _compute_c4(x0)
    if c4_0 < best_c4:
        best_c4 = c4_0
        best_roots = x0.copy()

    # Phase 2: Full 12-DOF L-BFGS-B refinement from alpha-optimal point
    if time_ok():
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: Differential evolution for global search - two passes with different seeds
    for de_seed in [42, 137]:
        if not time_ok():
            break
        bounds = [(1e-4, 150.0)] * 12
        try:
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                seed=de_seed,
                maxiter=300,
                popsize=20,
                tol=1e-10,
                polish=False,
                mutation=(0.3, 1.0),
                recombination=0.8,
                update_interval=10
            )
            if np.all(de_result.x > 0) and np.isfinite(de_result.fun):
                candidate = np.sort(de_result.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 4: Multi-start with perturbed initializations
    if time_ok():
        rng = np.random.default_rng(123)
        for scale in [0.1, 0.2, 0.35]:
            for _ in range(2):
                if not time_ok():
                    break
                pert = best_roots * (1.0 + scale * rng.standard_normal(12))
                pert = np.maximum(pert, 1e-3)
                pert = np.sort(pert)
                try:
                    res = minimize(
                        _objective_grad,
                        pert,
                        jac=True,
                        method='L-BFGS-B',
                        bounds=[(1e-8, None)] * 12,
                        options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
                    )
                    if np.all(res.x > 0) and np.isfinite(res.fun):
                        candidate = np.sort(res.x)
                        c4 = _compute_c4(candidate)
                        if c4 < best_c4:
                            best_c4 = c4
                            best_roots = candidate.copy()
                except Exception:
                    pass

    # Phase 5: Try neighboring alpha values and refine
    if time_ok():
        for alpha_offset in [0.5, 2.0, 0.1, 5.0, 0.2, 0.8]:
            if not time_ok():
                break
            for sign in [-1, 1]:
                a_try = best_alpha * alpha_offset if sign > 0 else best_alpha / alpha_offset
                if a_try < 1e-3 or a_try > 1e4:
                    continue
                try:
                    xg2, _ = roots_genlaguerre(12, a_try)
                    xg2 = np.sort(xg2)
                    if not np.all(xg2 > 0):
                        continue
                    res2 = minimize(
                        _objective_grad,
                        xg2,
                        jac=True,
                        method='L-BFGS-B',
                        bounds=[(1e-8, None)] * 12,
                        options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
                    )
                    if np.all(res2.x > 0) and np.isfinite(res2.fun):
                        candidate = np.sort(res2.x)
                        c4 = _compute_c4(candidate)
                        if c4 < best_c4:
                            best_c4 = c4
                            best_roots = candidate.copy()
                except Exception:
                    pass

    # Phase 6: Final high-precision polish: multiple passes with tightening tolerances
    for polish_iter in range(3):
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * 12,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 30}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
                else:
                    break  # No improvement, stop polishing
        except Exception:
            break

    # Final safety check and return
    if not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        # Fallback to standard Laguerre roots if something went wrong
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
