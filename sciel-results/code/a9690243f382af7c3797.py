"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a product of 1D Fourier-like trigonometric terms.
    
    The model is a linear combination of 10 separable trigonometric product terms:
      f(x, y) = sum_{i=0}^{9} p_i * phi_i(x) * psi_i(y)
    
    Basis functions:
      i=0: 1
      i=1: sin(x)
      i=2: cos(x)
      i=3: sin(y)
      i=4: cos(y)
      i=5: sin(x) * sin(y)
      i=6: sin(x) * cos(y)
      i=7: cos(x) * sin(y)
      i=8: cos(x) * cos(y)
      i=9: cos(2x)
    
    This basis prioritizes 1st-order interactions between x and y, which are often
    present in symbolic regression targets, while retaining low-frequency marginals
    and one higher-frequency term.
    """
    p = params
    
    # Precompute trig functions
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_y = np.sin(y)
    cos_y = np.cos(y)
    
    # Build basis terms
    term0 = np.ones_like(x)
    term1 = sin_x
    term2 = cos_x
    term3 = sin_y
    term4 = cos_y
    term5 = sin_x * sin_y
    term6 = sin_x * cos_y
    term7 = cos_x * sin_y
    term8 = cos_x * cos_y
    term9 = np.cos(2.0 * x)
    
    # Combine with parameters
    prediction = (
        p[0] * term0 +
        p[1] * term1 +
        p[2] * term2 +
        p[3] * term3 +
        p[4] * term4 +
        p[5] * term5 +
        p[6] * term6 +
        p[7] * term7 +
        p[8] * term8 +
        p[9] * term9
    )
    
    return prediction
# EVOLVE-BLOCK-END
