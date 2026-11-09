"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _legendre_basis(x: np.ndarray, degree: int = 9) -> np.ndarray:
    """
    Compute Legendre polynomial basis P_0(x) through P_degree(x)
    for x in [-3, 3], mapped to t in [-1, 1].
    
    Legendre polynomials are orthogonal with respect to the uniform measure,
    which matches the uniform distribution of the input data. This typically
    leads to a better-conditioned Gram matrix than Chebyshev polynomials
    for uniformly distributed points.
    
    Recurrence:
    P_0(t) = 1
    P_1(t) = t
    (n+1)P_{n+1}(t) = (2n+1)tP_n(t) - nP_{n-1}(t)
    """
    t = x / 3.0  # Map [-3, 3] to [-1, 1]
    
    n = len(x)
    basis = np.zeros((n, degree + 1))
    
    basis[:, 0] = 1.0
    if degree >= 1:
        basis[:, 1] = t
    
    for k in range(2, degree + 1):
        # (k) * P_k(t) = (2k-1)t * P_{k-1}(t) - (k-1) * P_{k-2}(t)
        basis[:, k] = ((2.0 * k - 1.0) * t * basis[:, k-1] - (k - 1.0) * basis[:, k-2]) / k
    
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a degree-9 Legendre polynomial basis.
    
    The input x is mapped from [-3, 3] to [-1, 1].
    params[0:10] are the coefficients for P_0 through P_9.
    
    Legendre polynomials are optimal for uniform data distribution,
    providing a well-conditioned linear system for the optimizer.
    """
    start_time = time.time()
    
    # Ensure x is 1D
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Compute Legendre basis (degree 9, so 10 basis functions)
    basis = _legendre_basis(x, degree=9)
    
    # params should have at least 10 elements
    coeffs = np.asarray(params, dtype=np.float64).ravel()[:10]
    
    # Linear combination: prediction = basis @ coeffs
    prediction = basis @ coeffs
    
    # Ensure finite output
    if not np.all(np.isfinite(prediction)):
        # Fallback to a simple linear fit if something goes wrong
        prediction = coeffs[0] * x + coeffs[1]
        
    # Deadline guard (safety margin)
    if time.time() - start_time > 110.0:
         # If we are somehow slow (unlikely for linear model), return best effort
         pass

    return prediction
# EVOLVE-BLOCK-END
