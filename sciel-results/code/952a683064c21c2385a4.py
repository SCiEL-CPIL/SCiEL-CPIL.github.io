"""Rewritten program: Chebyshev-basis degree-5 polynomial fit with numerical safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _chebyshev_to_monomial(cheb_coeffs, degree=5):
    """
    Convert Chebyshev coefficients (ascending in T_k) to monomial coefficients (ascending in x^k).
    Uses the recurrence for T_k in terms of x:
      T_0(x) = 1
      T_1(x) = x
      T_{k}(x) = 2*x*T_{k-1}(x) - T_{k-2}(x)
    We build each T_k as a monomial polynomial and accumulate.
    """
    n = degree + 1
    # monomials[k] = coefficients of T_k(x) in ascending powers of x, length k+1
    monomials = []
    monomials.append(np.array([1.0]))          # T_0
    monomials.append(np.array([0.0, 1.0]))     # T_1
    for k in range(2, n):
        # T_k = 2*x*T_{k-1} - T_{k-2}
        prev = monomials[k-1]
        prev2 = monomials[k-2]
        # 2*x*T_{k-1}: shift prev by 1 and multiply by 2
        shifted = np.zeros(k + 1)
        shifted[1:] = 2.0 * prev
        # subtract T_{k-2}
        shifted[:len(prev2)] -= prev2
        monomials.append(shifted)
    
    # Accumulate: sum cheb_coeffs[k] * T_k(x)
    result = np.zeros(degree + 1)
    for k in range(n):
        result += cheb_coeffs[k] * monomials[k]
    return result


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    Uses Chebyshev basis for numerical stability, then converts to monomial coefficients.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    if len(x) == 0 or len(y) == 0:
        return np.zeros(6)
    
    # Determine the range of x to map to [-1, 1] for Chebyshev fitting
    x_min = x.min()
    x_max = x.max()
    
    if x_max - x_min < 1e-12:
        # Degenerate case: constant x
        coeff = np.zeros(6)
        coeff[0] = y.mean()
        return coeff
    
    # Map x to t in [-1, 1]
    t = (2.0 * x - (x_max + x_min)) / (x_max - x_min)
    
    # Build Chebyshev Vandermonde matrix V where V[i,k] = T_k(t_i)
    n = len(x)
    degree = 5
    V = np.zeros((n, degree + 1))
    V[:, 0] = 1.0
    if degree >= 1:
        V[:, 1] = t
    for k in range(2, degree + 1):
        V[:, k] = 2.0 * t * V[:, k-1] - V[:, k-2]
    
    # Solve least squares: minimize ||V @ c - y||^2
    try:
        cheb_coeffs, residuals, rank, sv = np.linalg.lstsq(V, y, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback: simple linear fit
        coeff = np.zeros(6)
        coeff[:2] = np.polyfit(x, y, 1)
        # np.polyfit returns descending powers, so flip
        coeff[:2] = np.array([coeff[1], coeff[0]])
        return coeff
    
    # Convert Chebyshev coefficients to monomial coefficients in x
    # We need to express T_k(t) where t = (2x - (x_max+x_min))/(x_max-x_min)
    # Let a = 2/(x_max-x_min), b = -(x_max+x_min)/(x_max-x_min)
    # t = a*x + b
    a = 2.0 / (x_max - x_min)
    b = -(x_max + x_min) / (x_max - x_min)
    
    # T_k(t) where t = a*x + b
    # We need to expand T_k(a*x + b) in powers of x
    # Use the same recurrence but with t = a*x + b
    # T_0(t) = 1
    # T_1(t) = t = a*x + b
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    # Build monomial representations of T_k(t) where t = a*x + b
    # Represent each as a polynomial in x: coeffs ascending
    poly_T = []
    poly_T.append(np.array([1.0]))  # T_0 = 1
    poly_T.append(np.array([b, a]))  # T_1 = a*x + b
    
    for k in range(2, degree + 1):
        # T_k = 2*t*T_{k-1} - T_{k-2}
        # t * T_{k-1}: multiply polynomial T_{k-1} by (a*x + b)
        prev = poly_T[k-1]
        prev2 = poly_T[k-2]
        # t * prev = a*x*prev + b*prev
        # a*x*prev: shift prev by 1, multiply by a
        ax_prev = np.zeros(len(prev) + 1)
        ax_prev[1:] = a * prev
        # b*prev
        b_prev = b * prev
        # Sum: t * prev
        t_prev = ax_prev + b_prev
        # Subtract T_{k-2}
        result_poly = t_prev.copy()
        result_poly[:len(prev2)] -= prev2
        poly_T.append(result_poly)
    
    # Now sum cheb_coeffs[k] * T_k(t) expressed in monomials of x
    monomial_coeffs = np.zeros(degree + 1)
    for k in range(degree + 1):
        monomial_coeffs += cheb_coeffs[k] * poly_T[k]
    
    # Ensure finite values
    if not np.all(np.isfinite(monomial_coeffs)):
        # Fallback to linear
        coeff = np.zeros(6)
        try:
            c1, c0 = np.polyfit(x, y, 1)
            coeff[0] = c0
            coeff[1] = c1
        except:
            coeff[0] = y.mean()
        return coeff
    
    return monomial_coeffs
# EVOLVE-BLOCK-END
