import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_genlaguerre, roots_laguerre
from numpy.polynomial.polynomial import polyfromroots, polymul

_START = time.time()
_DEADLINE = 600.0
_MARGIN = 45.0


def _time_left():
    return _DEADLINE - (time.time() - _START) - _MARGIN


# Precompute factorials up to 80 (degree of P^2 is 48, moments up to that)
_FACTS = np.array([math.factorial(k) for k in range(80)], dtype=np.float64)


def _poly_moment(coeffs):
    """Compute integral of poly (ascending coeff) * e^{-x} dx = sum_k coeffs[k]*k!."""
    n = min(len(coeffs), len(_FACTS))
    if n == 0:
        return 0.0
    return float(np.dot(coeffs[:n], _FACTS[:n]))


def _compute_c4_and_grad_x(roots):
    """Compute C4 and gradient w.r.t roots x_i.
    P = prod (x-x_i)^2.
    Returns (c4, grad_x)"""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf, np.zeros_like(roots)

    # Build B(x) = prod (x - x_i), then P = B^2, P2 = P^2 = B^4
    B = polyfromroots(roots)  # ascending order, degree 12
    P = polymul(B, B)         # degree 24
    P2 = polymul(P, P)        # degree 48

    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        return np.inf, np.zeros_like(roots)
    
    c4 = I2 / (I1 * I1)
    
    # Analytic gradient w.r.t x_i
    # P = B^2, dP/dx_i = 2*B * dB/dx_i where dB/dx_i = -R_i, R_i = prod_{j!=i}(x-x_j)
    # dI1/dx_i = -2 * int B*R_i * e^{-x} dx
    # P2 = B^4, dP2/dx_i = 4*B^3 * dB/dx_i = -4*B^3*R_i
    # dI2/dx_i = -4 * int B^3*R_i * e^{-x} dx
    # dC4/dx_i = dI2/(I1^2) - 2*I2*dI1/(I1^3)
    
    # Precompute B^3 once (degree 36)
    B3 = polymul(polymul(B, B), B)
    
    grad_x = np.zeros_like(roots)
    for i in range(n):
        # R_i = prod_{j!=i}(x-x_j)
        R_i = np.array([1.0])
        for j in range(n):
            if j == i:
                continue
            R_i = polymul(R_i, np.array([-roots[j], 1.0]))
        
        # dI1/dx_i = -2 * int B*R_i * e^{-x} dx
        BR_i = polymul(B, R_i)  # degree 12 + 11 = 23
        dI1_i = -2.0 * _poly_moment(BR_i)
        
        # dI2/dx_i = -4 * int B^3 * R_i * e^{-x} dx
        B3R_i = polymul(B3, R_i)  # degree 47
        dI2_i = -4.0 * _poly_moment(B3R_i)
        
        grad_x[i] = dI2_i / (I1 * I1) - 2.0 * I2 * dI1_i / (I1 ** 3)
    
    return float(c4), grad_x


def _objective_y(y):
    """Objective function for log-roots y. x = exp(y)."""
    x = np.exp(y)
    c4, _ = _compute_c4_and_grad_x(x)
    return c4

def _objective_grad_y(y):
    """Objective and gradient for log-roots y. x = exp(y).
    grad_y = x * grad_x"""
    x = np.exp(y)
    c4, grad_x = _compute_c4_and_grad_x(x)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(y)
    grad_y = x * grad_x
    return c4, grad_y


def construct():
    m = 12
    n_roots = 12
    
    best_c4 = np.inf
    best_y = None
    best_x = None

    # Helper to update best
    def try_update(x_candidate):
        nonlocal best_c4, best_y, best_x
        x_candidate = np.sort(x_candidate)
        if np.any(x_candidate <= 0) or not np.all(np.isfinite(x_candidate)):
            return
        c4_val, _ = _compute_c4_and_grad_x(x_candidate)
        if c4_val < best_c4:
            best_c4 = c4_val
            best_x = x_candidate.copy()
            best_y = np.log(x_candidate)

    # Phase 1: Generate initial candidates
    # 1. Standard Laguerre (alpha=0)
    try:
        x0, _ = roots_laguerre(12)
        try_update(x0)
    except Exception:
        pass

    # 2. Generalized Laguerre for various alpha
    # We can try a grid of alphas
    alphas_to_try = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, -0.5, -0.9]
    for alpha in alphas_to_try:
        if _time_left() < 10: break
        try:
            xg, _ = roots_genlaguerre(12, alpha)
            xg = np.sort(xg)
            if np.all(xg > 0):
                try_update(xg)
        except Exception:
            pass

    # 3. Uniform log-spaced roots
    if _time_left() > 10:
        y_uniform = np.linspace(-1, 5, 12) # x from e^-1 to e^5
        x_uniform = np.exp(y_uniform)
        try_update(x_uniform)

    # Phase 2: Optimization in log-space
    # L-BFGS-B on y
    if best_y is not None:
        # Start with the best initial guess
        current_best_y = best_y.copy()
        
        # Run optimization
        # We can run multiple optimizations starting from different points
        starts = [current_best_y]
        
        # Add perturbed starts
        rng = np.random.default_rng(42)
        for i in range(5):
            if _time_left() < 20: break
            pert = current_best_y + 0.2 * rng.standard_normal(12)
            starts.append(pert)
            
        # Also add starts from other alpha values if they were good
        # (Already handled by try_update updating best_y if better)
        
        for y_start in starts:
            if _time_left() < 30: break
            try:
                res = minimize(
                    _objective_grad_y,
                    y_start,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50}
                )
                if np.isfinite(res.fun):
                    x_res = np.exp(res.x)
                    try_update(x_res)
            except Exception:
                pass

    # Phase 3: Further refinement and multi-start from the best found
    if best_y is not None:
        current_best_y = best_y.copy()
        
        # More aggressive multi-start
        rng = np.random.default_rng(123)
        for trial in range(10):
            if _time_left() < 40: break
            # Random perturbation in log space
            scale = 0.1 + 0.1 * trial
            pert = current_best_y + scale * rng.standard_normal(12)
            
            try:
                res = minimize(
                    _objective_grad_y,
                    pert,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
                )
                if np.isfinite(res.fun):
                    x_res = np.exp(res.x)
                    try_update(x_res)
            except Exception:
                pass

    # Phase 4: Final polish on the absolute best
    if best_y is not None:
        if _time_left() > 20:
            try:
                res = minimize(
                    _objective_grad_y,
                    best_y,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100}
                )
                if np.isfinite(res.fun):
                    x_res = np.exp(res.x)
                    try_update(x_res)
            except Exception:
                pass

    # Fallback
    if best_x is None:
        best_x = np.array([4.0 * (i + 1) for i in range(n_roots)], dtype=float)

    return [float(r) for r in best_x]
