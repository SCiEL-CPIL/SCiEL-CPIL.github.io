"""Symbolic regression with 1 input(s) - Rich basis with local features."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Linear combination of 10 basis functions on [-2, 2].
    
    Basis functions designed to capture a wide range of smooth functions:
      0: 1 (constant)
      1: x
      2: x^2
      3: x^3
      4: x^4
      5: tanh(x)
      6: tanh(2*x)
      7: tanh(4*x)
      8: exp(-x^2)  (Gaussian centered at 0)
      9: exp(-(x-1)^2) + exp(-(x+1)^2)  (Gaussians at ±1)
    
    The Gaussian terms allow capturing local peaks/valleys that polynomials
    and tanh cannot easily represent.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Polynomial terms
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = x ** 4
    
    # tanh non-linearities at different scales
    x_clipped = np.clip(x, -50, 50)
    b5 = np.tanh(x_clipped)
    b6 = np.tanh(2.0 * x_clipped)
    b7 = np.tanh(4.0 * x_clipped)
    
    # Gaussian terms for local features
    b8 = np.exp(-x ** 2)
    b9 = np.exp(-(x - 1.0) ** 2) + np.exp(-(x + 1.0) ** 2)
    
    # Linear combination
    y = (params[0] * b0 + params[1] * b1 + params[2] * b2 +
         params[3] * b3 + params[4] * b4 +
         params[5] * b5 + params[6] * b6 + params[7] * b7 +
         params[8] * b8 + params[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
