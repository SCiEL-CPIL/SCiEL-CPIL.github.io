"""Tensor-product Fourier basis for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a tensor-product Fourier (trigonometric) basis.
    This uses sine and cosine terms with distinct integer frequencies for both x and y,
    providing an orthogonal basis that is well-suited for smooth functions on [-1, 1].
    
    Basis terms (10 total):
    0: 1 (constant)
    1: cos(pi * x)
    2: cos(pi * y)
    3: sin(pi * x)
    4: sin(pi * y)
    5: cos(2*pi * x)
    6: cos(2*pi * y)
    7: sin(2*pi * x)
    8: sin(2*pi * y)
    9: cos(pi * x) * cos(pi * y)  (interaction term)
    """
    # Compute trigonometric basis functions
    c1x = np.cos(np.pi * x)
    s1x = np.sin(np.pi * x)
    c2x = np.cos(2.0 * np.pi * x)
    s2x = np.sin(2.0 * np.pi * x)
    
    c1y = np.cos(np.pi * y)
    s1y = np.sin(np.pi * y)
    c2y = np.cos(2.0 * np.pi * y)
    s2y = np.sin(2.0 * np.pi * y)
    
    # Interaction term
    c1xy = c1x * c1y
    
    # Linear combination of basis terms
    pred = (params[0] * 1.0 +
            params[1] * c1x +
            params[2] * c1y +
            params[3] * s1x +
            params[4] * s1y +
            params[5] * c2x +
            params[6] * c2y +
            params[7] * s2x +
            params[8] * s2y +
            params[9] * c1xy)
    
    return pred
# EVOLVE-BLOCK-END
