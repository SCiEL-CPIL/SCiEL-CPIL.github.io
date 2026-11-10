"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _chebyshev_basis(x, degree):
    """Return array of Chebyshev polynomials T_0(x) ... T_degree(x).
    
    x should be in [-1, 1] for standard Chebyshev. We'll map from [-2,2].
    """
    n = degree + 1
    T = np.zeros((n,) + x.shape, dtype=np.float64)
    T[0] = 1.0
    if degree >= 1:
        T[1] = x
    for k in range(2, n):
        T[k] = 2.0 * x * T[k-1] - T[k-2]
    return T

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Product-of-polynomials model: f(x) ≈ P_5(x) * Q_4(x)
    
    10 parameters: 6 for P (degree 5) and 4 for Q (degree 3),
    plus one shared constant, or simply 5+5 split.
    
    Actually: P has coefficients a_0..a_4 (5 params, degree 4)
    Q has coefficients b_0..b_4 (5 params, degree 4)
    Total: 10 params, product gives degree 8 polynomial.
    
    Uses Chebyshev basis for numerical stability.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a simple fallback: just the first param as constant
        return np.full_like(x, float(params[0]))
    
    # Map x from [-2, 2] to [-1, 1] for Chebyshev stability
    t = x * 0.5
    
    # Build Chebyshev basis up to degree 4 (5 basis functions)
    T = _chebyshev_basis(t, 4)  # shape (5, ...)
    
    # P(t) = sum_{i=0}^{4} p[i] * T_i(t)
    # Q(t) = sum_{i=0}^{4} p[5+i] * T_i(t)
    p = params
    
    P = p[0] * T[0] + p[1] * T[1] + p[2] * T[2] + p[3] * T[3] + p[4] * T[4]
    Q = p[5] * T[0] + p[6] * T[1] + p[7] * T[2] + p[8] * T[3] + p[9] * T[4]
    
    result = P * Q
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
