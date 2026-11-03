import time
import math
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.special import roots_laguerre

# Precompute factorials for moment computation
# P has degree 24 (12 double roots), P^2 has degree 48
_FACTS = np.array([math.factorial(k) for k in range(60)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly * e^{-x} dx over [0,inf).
    coeffs[k] is coefficient of x^k. Result = sum_k coeffs[k] * k!."""
    n = min(len(coeffs), 60)
    if n == 0:
        return 0.0
    return float(np.dot(coeffs[:n], _FACTS[:n]))

def _build_p_squared_coeffs(roots):
    """Given 12 roots, build P(x) = prod (x-x_i)^2 and return P2_coeffs = P^2.
    Coefficients in ascending power order."""
    from numpy.polynomial.polynomial import polyfromroots, polymul
    B = polyfromroots(roots)  # B = prod (x - x_i), degree 12
    P = polymul(B, B)          # P = B^2 = prod (x-x_i)^2, degree 24
    P2 = polymul(P, P)         # P^2, degree 48
    return P, P2

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx.
    P = prod (x-x_i)^2. Returns (c4, grad) or just c4."""
    roots = np.asarray(roots, dtype=np.float64)
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
    
    # Gradient computation
    grad = np.zeros_like(roots)
    for i in range(len(roots)):
        # R_i = prod_{j!=i} (x - x_j)^2
        R_i = np.array([1.0])
        for j in range(len(roots)):
            if j == i:
                continue
            R_i = np.polynomial.polynomial.polymul(
                R_i, np.polynomial.polynomial.polyfromroots([roots[j], roots[j]])
            )
        # dI1/dx_i = -2 * int (x-x_i)*R_i * e^{-x} dx
        xR_i = np.polynomial.polynomial.polymul(R_i, np.array([0.0, 1.0]))
        term1 = _poly_moment(xR_i) - roots[i] * _poly_moment(R_i)
        dI1_i = -2.0 * term1
        # dI2/dx_i = -4 * int (x-x_i)*R_i*P * e^{-x} dx
        RP = np.polynomial.polynomial.polymul(R_i, P)
        xRP = np.polynomial.polynomial.polymul(RP, np.array([0.0, 1.0]))
        term2 = _poly_moment(xRP) - roots[i] * _poly_moment(RP)
        dI2_i = -4.0 * term2
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
    SAFETY = 50.0
    deadline = start + DEADLINE - SAFETY

    def time_ok():
        return (time.time() - start) < deadline

    # Initial guess: standard Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    best_roots = x0.copy()
    best_c4 = _compute_c4(best_roots)

    # First, do a quick L-BFGS-B polish on the Laguerre roots
    try:
        res = minimize(
            _objective_grad,
            best_roots,
            jac=True,
            method='L-BFGS-B',
            bounds=[(1e-6, None)] * 12,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40}
        )
        if np.all(res.x > 0) and np.isfinite(res.fun):
            candidate = np.sort(res.x)
            c4 = _compute_c4(candidate)
            if c4 < best_c4:
                best_c4 = c4
                best_roots = candidate.copy()
    except Exception:
        pass

    # Use BasinHopping to explore the landscape
    if time_ok():
        def step(x):
            # Random perturbation
            # Scale of perturbation based on current root magnitudes
            scale = np.mean(x) * 0.1
            return x + np.random.normal(0, scale, len(x))

        # We use L-BFGS-B as the local minimizer inside BasinHopping
        # Note: BasinHopping works with a single objective function, not (fun, grad)
        # So we need to wrap it.
        
        def bh_objective(x):
            return _compute_c4(x)

        # BasinHopping parameters
        # maxfun: max function evaluations
        # We don't know how long one eval takes, but it's fast (<1ms).
        # Let's allow many evaluations.
        try:
            bh_result = basinhopping(
                bh_objective,
                best_roots,
                minimizer_kwargs={
                    'method': 'L-BFGS-B',
                    'jac': _objective_grad,
                    'bounds': [(1e-6, None)] * 12,
                    'options': {'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                },
                step=step,
                niter=100, # Number of basin hops
                T=1.0, # Temperature
                accept_test=lambda x, fx, x_new, fx_new: fx_new < fx, # Strictly accept improvements
                callback=None
            )
            
            if np.all(bh_result.x > 0) and np.isfinite(bh_result.fval):
                candidate = np.sort(bh_result.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception as e:
            pass

    # Final high-precision polish
    if time_ok():
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * 12,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    return [float(xi) for xi in best_roots]
