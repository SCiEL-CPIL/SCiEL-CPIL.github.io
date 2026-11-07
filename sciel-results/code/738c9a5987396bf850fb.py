"""Symbolic regression: equation(x, params) with 10 free params."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using 10 parameters.
    Uses a mix of scaled polynomials and localized Gaussian basis functions.
    This structure is linear in params, allowing BFGS to converge efficiently
    to the optimal linear combination for the fixed basis.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if x.ndim == 0:
        x = x.reshape(1)

    # Scale x from [0.1, 6] to [-1, 1] for better numerical conditioning
    # Midpoint: 3.05, Half-range: 2.95
    t = (x - 3.05) / 2.95
    
    try:
        # Basis functions
        # 1-3: Polynomials (global trend)
        phi0 = 1.0
        phi1 = t
        phi2 = t**2
        
        # 4-10: Gaussian RBFs (local features)
        # Width parameter sigma=0.5 in t-space (so exp(-2*t^2) roughly)
        # Let's use exp(-4 * (t - c)^2) for good locality
        phi3 = np.exp(-4.0 * (t - 0.0)**2)
        phi4 = np.exp(-4.0 * (t - 0.5)**2)
        phi5 = np.exp(-4.0 * (t + 0.5)**2)
        phi6 = np.exp(-4.0 * (t - 1.0)**2)
        phi7 = np.exp(-4.0 * (t + 1.0)**2)
        phi8 = np.exp(-4.0 * (t - 0.25)**2)
        phi9 = np.exp(-4.0 * (t + 0.25)**2)
        
        result = (p[0] * phi0 +
                  p[1] * phi1 +
                  p[2] * phi2 +
                  p[3] * phi3 +
                  p[4] * phi4 +
                  p[5] * phi5 +
                  p[6] * phi6 +
                  p[7] * phi7 +
                  p[8] * phi8 +
                  p[9] * phi9)
        
        # Check for non-finite values
        if not np.all(np.isfinite(result)):
            return np.zeros_like(x, dtype=np.float64)
            
        return result
    except Exception:
        return np.zeros_like(x, dtype=np.float64)
# EVOLVE-BLOCK-END
