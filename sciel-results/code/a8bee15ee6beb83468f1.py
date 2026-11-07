"""Symbolic regression: equation(x, params) with 10 free params."""
# EVOLVE-BLOCK-START
import numpy as np
import time

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
        
        T = np.ones_like(t)
        if len(t) > 0:
            T1 = t.copy()
            
            # Build the basis matrix
            basis = np.zeros((len(t), 10))
            basis[:, 0] = 1.0  # T0
            if len(t) > 0:
                basis[:, 1] = T1  # T1
                
                # Compute T2 through T9
                for n in range(2, 10):
                    Tn = 2.0 * t * T1 - T  # Tn = 2*x*Tn-1 - Tn-2
                    basis[:, n] = Tn
                    T, T1 = T1, Tn
        
        # Linear combination of basis functions
        result = basis @ p
        
        # Check for non-finite values
        if not np.all(np.isfinite(result)):
            return np.zeros_like(x, dtype=np.float64)
            
        return result
    except Exception:
        return np.zeros_like(x, dtype=np.float64)
# EVOLVE-BLOCK-END
