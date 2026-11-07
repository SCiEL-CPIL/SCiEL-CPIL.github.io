"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y from x using a linear combination of 10 well-conditioned basis functions.
    The basis is chosen to cover a wide range of smooth functions on [-3, 3]:
    - Low-degree polynomials for non-oscillatory trends
    - Trig functions for periodic behavior
    - Gaussian-like terms for localized features
    """
    # Ensure x is a 1D float array
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    
    # Basis functions
    # 0: constant
    b0 = np.ones_like(x)
    
    # 1: linear
    b1 = x
    
    # 2: quadratic
    b2 = x * x
    
    # 3: cubic
    b3 = x * x * x
    
    # 4: sin(x)
    b4 = np.sin(x)
    
    # 5: cos(x)
    b5 = np.cos(x)
    
    # 6: sin(2x)
    b6 = np.sin(2.0 * x)
    
    # 7: cos(2x)
    b7 = np.cos(2.0 * x)
    
    # 8: Gaussian centered at 0
    b8 = np.exp(-0.5 * x * x)
    
    # 9: Inverse quadratic (Lorentzian-like)
    b9 = 1.0 / (1.0 + x * x)
    
    # Linear combination
    y = (p[0] * b0 +
         p[1] * b1 +
         p[2] * b2 +
         p[3] * b3 +
         p[4] * b4 +
         p[5] * b5 +
         p[6] * b6 +
         p[7] * b7 +
         p[8] * b8 +
         p[9] * b9)
    
    return y

# EVOLVE-BLOCK-END
