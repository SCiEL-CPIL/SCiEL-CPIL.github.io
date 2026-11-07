"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fits a function using a Chebyshev polynomial basis of degree 9.
    The input x is scaled to [-1, 1] to utilize the stability of Chebyshev polynomials.
    params[0..9] are the coefficients for T_0(t) ... T_9(t).
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=np.float64)
    
    # Scale x from [-3, 3] to [-1, 1]
    # This maps the domain to the natural domain of Chebyshev polynomials
    t = x / 3.0
    
    # Clamp t to [-1, 1] just in case of numerical drift or out-of-bounds inputs
    # Though the problem states inputs are from [-3, 3], clamping prevents NaNs in edge cases
    t = np.clip(t, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    basis = np.empty((len(t), 10), dtype=np.float64)
    basis[:, 0] = 1.0
    if len(t) > 0:
        basis[:, 1] = t
        for k in range(2, 10):
            basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
            
    # Linear combination of basis functions
    # params is shape (10,), basis is shape (N, 10)
    # Result is shape (N,)
    y_pred = basis @ params
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(y_pred)):
        # If numerical instability occurs, fall back to a simple linear model
        # This is a defensive measure, unlikely to trigger with Chebyshev
        return params[0] * x + params[1]
        
    return y_pred
# EVOLVE-BLOCK-END
