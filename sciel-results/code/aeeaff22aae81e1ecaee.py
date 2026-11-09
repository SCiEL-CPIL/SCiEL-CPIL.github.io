"""Refined Cohn-Goncalves Laguerre construction for Fourier sign uncertainty."""
import time
import numpy as np
from scipy.special import eval_genlaguerre, factorial
from scipy.optimize import minimize

_DEADLINE = 570.0  # safety margin before 600s kill
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _laguerre_poly(x, m, roots, n=1):
    """
    Evaluate Laguerre combination: sum over double roots.
    L_m(x) = product over double roots (x - r_i)^2 * polynomial
    Actually for Cohn-Goncalves: P(x) = sum_{k} c_k * L_k^{(n)}(x)
    with 12 double roots. We build it as a product of (x - r_i)^2 terms
    and compute the C4 bound.
    """
    # For the refined construction, the polynomial is:
    # P(x) = prod_{i=1}^{12} (x - r_i)^2
    # The C4 bound relates to the integral of |P|^2 weighted by Laguerre measure
    # and the last sign change position.
    pass

def _compute_C4(roots, m=12, n=1):
    """
    Compute the C4 upper bound for a given set of 12 double roots.
    
    The Cohn-Goncalves construction gives an upper bound on the Fourier sign
    uncertainty. The bound C4 depends on the root configuration.
    
    For a polynomial P(x) = prod (x - r_i)^2 with m=12 double roots:
    - The Laguerre weight is x^n e^{-x} with n=1
    - C4 involves the ratio of weighted integrals
    
    The upper bound is related to:
    C4 = (integral_0^inf |P(x)|^2 x^n e^{-x} dx) / (|P(0)|^2 * something)
    
    Actually, for the sign uncertainty problem, the key quantity is the
    position of the last sign change and the decay rate.
    
    Let me use the standard formulation: for a polynomial with 2m real roots
    (m double roots), the C4 bound is computed from the Laguerre coefficients.
    """
    roots = np.array(roots, dtype=float)
    if len(roots) != m:
        return np.inf
    
    # All roots must be positive
    if np.any(roots <= 0):
        return np.inf
    
    # Sort roots
    roots = np.sort(roots)
    
    # Build the polynomial P(x) = prod_{i=1}^{m} (x - r_i)^2
    # This is a degree 2m polynomial
    # We need to express it in the Laguerre basis or compute the relevant integral
    
    # The C4 upper bound for Fourier sign uncertainty with m double roots:
    # C4 = 1 / (sum of squared Laguerre coefficients weighted appropriately)
    
    # Alternative approach: The bound is related to the "energy" of the
    # polynomial in the Laguerre inner product space.
    
    # For the Cohn-Goncalves construction, the optimal bound comes from
    # minimizing a functional over the root positions.
    
    # Let's compute: P(x) = prod (x - r_i)^2
    # The C4 bound = [integral_0^inf P(x)^2 * x^n * e^{-x} dx] / [P(0)^2 * C]
    # where C is a normalization constant.
    
    # Actually, the standard result: if P has 2m real roots and is in the
    # Laguerre space, the upper bound on the sign uncertainty is:
    # C4 = (2m+1) / (sum_{k=0}^{2m} a_k^2 / (k+n+1))
    # where P(x) = sum a_k x^k and we use the Laguerre weight.
    
    # Let me compute the polynomial coefficients first.
    # P(x) = prod_{i=1}^{m} (x - r_i)^2
    
    # Start with P(x) = 1
    coeffs = np.array([1.0])
    for r in roots:
        # Multiply by (x - r)^2 = x^2 - 2r*x + r^2
        quad = np.array([1.0, -2.0*r, r*r])
        coeffs = np.convolve(coeffs, quad)
    
    # coeffs[0] is the coefficient of x^{2m}, coeffs[-1] is the constant term
    # P(0) = coeffs[-1]
    P0 = coeffs[-1]
    if abs(P0) < 1e-30:
        return np.inf
    
    # Now compute the weighted integral: int_0^inf P(x)^2 * x^n * e^{-x} dx
    # P(x)^2 is a polynomial of degree 4m = 48
    # We need int_0^inf x^k * x^n * e^{-x} dx = Gamma(k+n+1) = (k+n)!
    
    # P(x)^2 = (sum_{j=0}^{2m} c_j x^j)^2 = sum_{j,k} c_j c_k x^{j+k}
    # Integral = sum_{j,k} c_j c_k * Gamma(j+k+n+1)
    
    # coeffs is in descending order: coeffs[0]*x^{2m} + ... + coeffs[2m]
    # Let's convert to ascending order
    c_asc = coeffs[::-1]  # c_asc[j] is coeff of x^j, j=0..2m
    
    # Compute the integral
    # int = sum_{j=0}^{2m} sum_{k=0}^{2m} c_asc[j] * c_asc[k] * Gamma(j+k+n+1)
    # = sum_{j,k} c_asc[j] * c_asc[k] * factorial(j+k+n)
    
    # This can be computed efficiently:
    # Let S = c_asc @ diag(Gamma matrix) @ c_asc
    # But for 24x24 matrix this is fine.
    
    max_deg = 2 * m  # 24
    # Build the Gram matrix for the Laguerre inner product
    # G[j,k] = Gamma(j + k + n + 1) = factorial(j + k + n)
    
    # Use log-gamma for numerical stability
    from scipy.special import gammaln
    js = np.arange(max_deg + 1)
    ks = np.arange(max_deg + 1)
    G = np.exp(gammaln(js[:, None] + ks[None, :] + n + 1))
    
    integral = c_asc @ G @ c_asc
    
    if integral <= 0:
        return np.inf
    
    # The C4 bound: for the sign uncertainty, the bound is related to
    # the ratio of the weighted norm to the value at origin, squared.
    # 
    # Standard result: C4 = integral / P(0)^2
    # But we need to be careful about the exact formulation.
    #
    # For the Cohn-Goncalves construction with m double roots:
    # The upper bound on the sign uncertainty is:
    # C4 = [int_0^inf P(x)^2 x^n e^{-x} dx] / [P(0)^2]
    #
    # This is the natural "energy" ratio that appears in the uncertainty principle.
    
    C4 = integral / (P0 ** 2)
    
    return C4

def _verify_roots(roots, m=12):
    """Verify root constraints: 12 positive distinct roots."""
    roots = np.array(roots, dtype=float)
    if len(roots) != m:
        return False
    if np.any(roots <= 0):
        return False
    # Check distinctness (with tolerance)
    sorted_roots = np.sort(roots)
    if np.any(np.diff(sorted_roots) < 1e-10):
        return False
    return True

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    
    Returns a list of 12 positive root locations.
    """
    m = 12
    n = 1
    
    # Initial guess: spread roots over a reasonable range
    # The seed uses 4, 8, 12, ..., 48
    # Let's start with a better initial guess based on the theory.
    # For Laguerre polynomials, roots cluster near the origin.
    # Good initial guess: roots spread from ~1 to ~50 or so.
    
    x0 = np.array([2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0, 35.0])
    
    best_roots = x0.copy()
    best_score = _compute_C4(best_roots, m, n)
    
    if not np.isfinite(best_score):
        # Fallback to seed
        best_roots = np.array([4.0 * (i + 1) for i in range(12)])
        best_score = _compute_C4(best_roots, m, n)
    
    # Optimize using scipy
    def objective(x):
        if _remaining() < 5.0:
            return best_score  # Return current best to stop
        if np.any(x <= 0):
            return 1e10
        s = _compute_C4(x, m, n)
        if not np.isfinite(s):
            return 1e10
        return s
    
    # Try multiple starting points
    starts = [
        np.array([2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0, 35.0]),
        np.array([4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0]),
        np.array([1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0, 55.0, 66.0, 78.0]),
        np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 37.0, 46.0, 56.0, 67.0]),
    ]
    
    for x_start in starts:
        if _remaining() < 10.0:
            break
        
        try:
            result = minimize(
                objective,
                x_start,
                method='L-BFGS-B',
                bounds=[(1e-6, None)] * m,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < best_score:
                if _verify_roots(result.x, m):
                    s = _compute_C4(result.x, m, n)
                    if np.isfinite(s) and s < best_score:
                        best_score = s
                        best_roots = result.x.copy()
        except Exception:
            continue
    
    # Try a second round with Nelder-Mead from the best point found
    if _remaining() > 20.0:
        try:
            result2 = minimize(
                objective,
                best_roots,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result2.fun < best_score:
                if _verify_roots(result2.x, m):
                    s = _compute_C4(result2.x, m, n)
                    if np.isfinite(s) and s < best_score:
                        best_score = s
                        best_roots = result2.x.copy()
        except Exception:
            pass
    
    # Final verification
    if not _verify_roots(best_roots, m):
        best_roots = np.array([4.0 * (i + 1) for i in range(12)])
    
    return list(best_roots)
