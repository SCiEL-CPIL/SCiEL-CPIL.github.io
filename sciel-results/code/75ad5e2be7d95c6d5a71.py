"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute Chebyshev basis for 200 points (training size)
# But we need to handle any number of points, so we compute on the fly
# However, for speed, we can precompute the Chebyshev polynomials

def _chebyshev_basis(x: np.ndarray, degree: int = 9) -> np.ndarray:
    """
    Compute Chebyshev polynomial basis T_0(x) through T_degree(x)
    for x in [-3, 3], mapped to t in [-1, 1].
    
    T_n(t) = cos(n * arccos(t))
    
    Using recurrence for numerical stability:
    T_0(t) = 1
    T_1(t) = t
    T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    """
    t = x / 3.0  # Map [-3, 3] to [-1, 1]
    
    n = len(x)
    basis = np.zeros((n, degree + 1))
    
    basis[:, 0] = 1.0
    basis[:, 1] = t
    
    for k in range(2, degree + 1):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
    
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a degree-9 Chebyshev polynomial basis.
    
    The input x is mapped from [-3, 3] to [-1, 1] for optimal Chebyshev conditioning.
    params[0:10] are the coefficients for T_0 through T_9.
    
    This provides a well-conditioned, globally smooth approximation that BFGS
    can efficiently optimize for smooth target functions.
    """
    # Ensure x is 1D
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Compute Chebyshev basis (degree 9, so 10 basis functions)
    basis = _chebyshev_basis(x, degree=9)
    
    # params should have at least 10 elements
    coeffs = np.asarray(params, dtype=np.float64).ravel()[:10]
    
    # Linear combination: prediction = basis @ coeffs
    prediction = basis @ coeffs
    
    # Ensure finite output
    if not np.all(np.isfinite(prediction)):
        # Fallback to a simple linear fit if something goes wrong
        prediction = coeffs[0] * x + coeffs[1]
    
    return prediction
# EVOLVE-BLOCK-END
