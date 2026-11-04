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
    
    # Fast gradient via logarithmic derivative
    # P = Q^2 where Q = prod_j (x - x_j)
    # dP/dx_i = 2Q * dQ/dx_i = 2Q * (-Q/(x - x_i)) = -2Q^2/(x - x_i) = -2P/(x - x_i)
    # dI1/dx_i = -2 * int P/(x-x_i) * e^{-x} dx
    # dI2/dx_i = 2 * int P * dP/dx_i * e^{-x} dx = -4 * int P^2/(x-x_i) * e^{-x} dx
    #
    # Compute S_i(x) = P(x)/(x - x_i) by synthetic division
    # Then int S_i e^{-x} dx and int x*S_i e^{-x} dx give us what we need.
    # P/(x-x_i) is a polynomial of degree 2n-1.
    # We need int P/(x-x_i) e^{-x} dx = sum_k c_k * k!  (ascending coeffs)
    # And int x*P/(x-x_i) e^{-x} dx = sum_k c_k * (k+1)!
    
    # Build Q = prod(x - x_j), then P = Q^2
    from numpy.polynomial.polynomial import polyfromroots, polymul
    Q = polyfromroots(roots)
    # P = Q^2 (already have it)
    
    # For each i, compute S_i = P / (x - x_i) via synthetic division
    # P has degree 2n, so S_i has degree 2n-1
    # Synthetic division of P by (x - r): 
    #   P(x) = (x - r) * S(x) + P(r)
    #   S(x) coefficients from highest to lowest:
    #   s_{2n-1} = p_{2n}
    #   s_{k} = p_{k+1} + r * s_{k+1}
    # In ascending order this is trickier. Let's use polynomial division.
    
    # Actually, let's use the fact that P = Q^2 and dQ/dx_i = -Q/(x-x_i)
    # So dP/dx_i = 2Q * (-Q/(x-x_i)) = -2Q^2/(x-x_i) = -2P/(x-x_i)
    # int dP/dx_i e^{-x} dx = -2 int P/(x-x_i) e^{-x} dx
    # int P * dP/dx_i e^{-x} dx = -2 int P^2/(x-x_i) e^{-x} dx
    # dI1/dx_i = -2 * int P/(x-x_i) e^{-x} dx
    # dI2/dx_i = 2 * int P * dP/dx_i e^{-x} dx = -4 * int P^2/(x-x_i) e^{-x} dx
    
    # To compute int P/(x-x_i) e^{-x} dx:
    # P/(x-x_i) is a polynomial. Let's compute it via numpy poly division.
    # Use np.polynomial.polynomial.polydiv for ascending order.
    
    from numpy.polynomial.polynomial import polydiv
    
    # P in ascending order, divisor is (x - x_i) = [-x_i, 1] in ascending
    divisor = np.array([-roots[i], 1.0])
    
    dI1 = np.zeros(n)
    dI2 = np.zeros(n)
    
    for i in range(n):
        # S_i = P / (x - x_i)
        S_i, rem = polydiv(P, divisor)
        # int S_i e^{-x} dx
        int_S = _poly_moment(S_i)
        # int x * S_i e^{-x} dx = sum c_k (k+1)!
        xS = np.polynomial.polynomial.polymul(S_i, np.array([0.0, 1.0]))
        int_xS = _poly_moment(xS)
        # dI1/dx_i = -2 * int P/(x-x_i) e^{-x} dx = -2 * int_S
        dI1[i] = -2.0 * int_S
        
        # For dI2: need int P^2/(x-x_i) e^{-x} dx
        T_i, rem2 = polydiv(P2, divisor)
        int_T = _poly_moment(T_i)
        dI2[i] = -4.0 * int_T
    
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

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 10.0  # Reduced safety margin to allow more work, but strict guard remains

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    best_roots = None
    best_c4 = np.inf
    best_alpha = 1.0

    # Phase 1: 1D search over alpha for generalized Laguerre nodes
    # This is fast and provides a strong initial guess
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

    # Phase 3: Multi-start L-BFGS-B from perturbed best
    # We skip 12D Differential Evolution as it is too slow.
    # Instead, we do multi-start from local perturbations which is faster and effective.
    if time_ok():
        candidates_to_refine = [best_roots.copy()]
        rng = np.random.default_rng(42)
        
        # Perturbations around best
        for scale in [0.02, 0.05, 0.1, 0.2]:
            for _ in range(2):
                pert = best_roots * (1.0 + scale * rng.standard_normal(12))
                pert = np.maximum(pert, 1e-3)
                pert = np.sort(pert)
                candidates_to_refine.append(pert)
        
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

    # Phase 4: Final high-precision L-BFGS-B polish
    for polish_iter in range(5):
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * 12,
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 50}
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
