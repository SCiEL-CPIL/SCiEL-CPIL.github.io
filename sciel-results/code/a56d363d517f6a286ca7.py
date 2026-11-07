"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use a degree-9 Chebyshev polynomial basis for symbolic regression.
    
    The Chebyshev polynomials T_k(x) form an orthogonal basis on [-1, 1] and
    are well-conditioned. We map x from [-2, 2] to [-1, 1] via u = x/2,
    then compute T_0(u) through T_9(u).
    
    params[0..9] are the coefficients for T_0 through T_9.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map x from [-2, 2] to [-1, 1]
    u = 0.5 * x
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    # T_0(u) = 1
    # T_1(u) = u
    # T_{k+1}(u) = 2*u*T_k(u) - T_{k-1}(u)
    
    T = np.zeros((10, len(x)), dtype=np.float64)
    T[0, :] = 1.0
    if len(x) > 0:
        T[1, :] = u
    for k in range(2, 10):
        T[k, :] = 2.0 * u * T[k-1, :] - T[k-2, :]
    
    # Compute result as linear combination
    result = np.zeros_like(x, dtype=np.float64)
    for k in range(10):
        result += params[k] * T[k, :]
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
