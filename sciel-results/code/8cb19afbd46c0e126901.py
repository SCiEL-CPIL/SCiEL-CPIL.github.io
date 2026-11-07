"""Symbolic regression: equation(x, params) with 10 free params."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using 10 parameters.
    Uses Chebyshev polynomials T0-T9, which are orthogonal on [-1,1]
    and well-conditioned for BFGS optimization from all-ones start.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if x.ndim == 0:
        x = x.reshape(1)

    # Scale x from [0.1, 6] to [-1, 1] for better numerical conditioning
    # Midpoint: 3.05, Half-range: 2.95
    t = (x - 3.05) / 2.95
    
    try:
        # Compute Chebyshev polynomials T0 through T9
        # T0(x) = 1
        # T1(x) = x
        # Tn(x) = 2*x*Tn-1(x) - Tn-2(x)
        
        n_points = len(x)
        T = np.zeros((10, n_points), dtype=np.float64)
        T[0, :] = 1.0
        T[1, :] = t
        
        for n in range(2, 10):
            T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
        
        # Linear combination: result = sum(p[i] * T[i, :])
        result = T @ p
        
        # Check for non-finite values
        if not np.all(np.isfinite(result)):
            return np.zeros_like(x, dtype=np.float64)
            
        return result
    except Exception:
        return np.zeros_like(x, dtype=np.float64)
# EVOLVE-BLOCK-END
