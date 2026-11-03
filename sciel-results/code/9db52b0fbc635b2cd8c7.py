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

def _build_p_squared_coeffs(roots):
    """Given 12 roots, build P(x) = prod (x-x_i)^2 and return (P_coeffs, P2_coeffs)."""
    from numpy.polynomial.polynomial import polyfromroots, polymul
    B = polyfromroots(roots)
    P = polymul(B, B)
    P2 = polymul(P, P)
    return P, P2

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
    
    # Gradient computation: dI1/dx_i and dI2/dx_i
    # P = prod_j (x - x_j)^2
    # dP/dx_i = -2(x - x_i) * prod_{j != i}(x - x_j)^2 = -2(x - x_i) * P / (x - x_i)^2
    # So dP/dx_i = -2 * P / (x - x_i) ... but this is a polynomial division
    # Better: R_i = prod_{j != i}(x - x_j)^2, then dP/dx_i = -2(x - x_i) R_i
    # I1 = int P e^{-x} dx, dI1/dx_i = int dP/dx_i e^{-x} dx = -2 int (x - x_i) R_i e^{-x} dx
    #    = -2 (int x R_i e^{-x} dx - x_i int R_i e^{-x} dx)
    grad = np.zeros_like(roots)
    for i in range(n):
        R_i = np.ones(1)
        for j in range(n):
            if j == i:
                continue
            R_i = np.polynomial.polynomial.polymul(R_i, np.polynomial.polynomial.polyfromroots([roots[j], roots[j]]))
        xR_i = np.polynomial.polynomial.polymul(R_i, np.array([0.0, 1.0]))
        int_R = _poly_moment(R_i)
        int_xR = _poly_moment(xR_i)
        dI1_i = -2.0 * (int_xR - roots[i] * int_R)
        
        # dI2/dx_i = 2 int P dP/dx_i e^{-x} dx = -4 int P (x - x_i) R_i e^{-x} dx
        RP = np.polynomial.polynomial.polymul(R_i, P)
        xRP = np.polynomial.polynomial.polymul(RP, np.array([0.0, 1.0]))
        int_RP = _poly_moment(RP)
        int_xRP = _poly_moment(xRP)
        dI2_i = -4.0 * (int_xRP - roots[i] * int_RP)
        
        grad[i] = dI2_i / (I1 * I1) - 2.0 * I2 * dI1_i / (I1 ** 3)
    
    return float(c4), grad

def _objective(roots):
    return _compute_c4(roots)

def _objective_grad(roots):
    c4, g = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(roots)
    return c4, g

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 45.0

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # Phase 1: Thorough 1D search over alpha for generalized Laguerre nodes
    # This reduces the 12-dim problem to 1-dim for a strong initial guess
    best_roots = None
    best_c4 = np.inf
    best_alpha = 1.0
    
    # Dense sweep over alpha
    alpha_values = np.concatenate([
        np.logspace(-2, 3, 50),      # 0.01 to 1000
        np.linspace(0.1, 50, 100),    # fine grid
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
    
    # Refine alpha around best using golden-section or fine grid
    if best_roots is not None and time_ok():
        a_lo = max(best_alpha * 0.5, 0.001)
        a_hi = best_alpha * 2.0
        # Fine grid around best
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
        # Even finer
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

    # Phase 2: L-BFGS-B refinement from the best alpha point
    if time_ok():
        try:
            res = minimize(
                _objective_grad,
                best_roots.copy(),
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-6, None)] * 12,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: Differential evolution for global search
    if time_ok():
        for de_seed in [42, 137, 256, 999, 7777]:
            if not time_ok():
                break
            bounds = [(1e-4, 500.0)] * 12
            try:
                de_result = differential_evolution(
                    _objective,
                    bounds=bounds,
                    seed=de_seed,
                    maxiter=150,
                    popsize=20,
                    tol=1e-10,
                    polish=False,
                    mutation=(0.4, 1.2),
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

    # Phase 4: Multi-start L-BFGS-B from perturbed best and nearby alpha values
    if time_ok():
        candidates_to_refine = [best_roots.copy()]
        rng = np.random.default_rng(42)
        
        # Perturbations around best
        for scale in [0.02, 0.05, 0.1, 0.2, 0.3]:
            for _ in range(3):
                pert = best_roots * (1.0 + scale * rng.standard_normal(12))
                pert = np.maximum(pert, 1e-3)
                pert = np.sort(pert)
                candidates_to_refine.append(pert)
        
        # Nearby alpha values
        for da in [0.05, 0.1, 0.2, 0.5]:
            for sign in [1, -1]:
                a_try = best_alpha * (1.0 + sign * da)
                if a_try > 0.001:
                    try:
                        xg, _ = roots_genlaguerre(12, float(a_try))
                        xg = np.sort(xg)
                        if np.all(xg > 0):
                            candidates_to_refine.append(xg.copy())
                    except Exception:
                        pass

        for init in candidates_to_refine:
            if not time_ok():
                break
            try:
                res = minimize(
                    _objective_grad,
                    init,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(1e-6, None)] * 12,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50}
                )
                if np.all(res.x > 0) and np.isfinite(res.fun):
                    candidate = np.sort(res.x)
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
            except Exception:
                pass

    # Phase 5: Final high-precision polish
    for polish_iter in range(8):
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * 12,
                options={'maxiter': 5000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
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
