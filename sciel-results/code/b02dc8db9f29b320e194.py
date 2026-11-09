"""Direct approach: Degree-9 Chebyshev polynomial basis with column normalization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a degree-9 Chebyshev polynomial basis
    with column normalization to improve conditioning for BFGS optimization.
    
    The 10 parameters correspond to coefficients of normalized Chebyshev polynomials
    T_0(u) to T_9(u), where u = x / 3.0 maps the input range [-3, 3] to [-1, 1].
    
    Column normalization (unit L2 norm over the domain) improves the condition number
    of the design matrix, helping BFGS converge more reliably from the all-ones start.
    
    Parameters:
    x : np.ndarray
        Input points.
    params : np.ndarray
        Array of 10 parameters to be fitted.
        
    Returns:
    np.ndarray
        Predicted values.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    
    if n == 0:
        return np.array([])
    
    # Scale x to [-1, 1]
    u = x / 3.0
    
    # Compute Chebyshev polynomials T_0(u) to T_9(u)
    # T_0(u) = 1
    # T_1(u) = u
    # T_k(u) = 2*u*T_{k-1}(u) - T_{k-2}(u)
    
    V = np.empty((n, 10), dtype=np.float64)
    V[:, 0] = 1.0
    if n > 0:
        V[:, 1] = u
        
        for k in range(2, 10):
            V[:, k] = 2.0 * u * V[:, k-1] - V[:, k-2]
    
    # Normalize columns to unit L2 norm for better conditioning
    # Use a fixed reference set for consistent normalization
    ref_u = np.linspace(-1.0, 1.0, 100)
    ref_V = np.empty((100, 10), dtype=np.float64)
    ref_V[:, 0] = 1.0
    ref_V[:, 1] = ref_u
    for k in range(2, 10):
        ref_V[:, k] = 2.0 * ref_u * ref_V[:, k-1] - ref_V[:, k-2]
    
    col_norms = np.linalg.norm(ref_V, axis=0)
    col_norms[col_norms < 1e-15] = 1.0
    V = V / col_norms
    
    return V @ params
# EVOLVE-BLOCK-END
