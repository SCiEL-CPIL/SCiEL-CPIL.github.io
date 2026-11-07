"""Truncated Fourier series basis for symbolic regression."""
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a truncated Fourier series basis.
    
    Basis functions:
      0: constant (1)
      1: cos(2*pi*1*x / 5.9)
      2: sin(2*pi*1*x / 5.9)
      3: cos(2*pi*2*x / 5.9)
      4: sin(2*pi*2*x / 5.9)
      5: cos(2*pi*3*x / 5.9)
      6: sin(2*pi*3*x / 5.9)
      7: cos(2*pi*4*x / 5.9)
      8: sin(2*pi*4*x / 5.9)
      9: cos(2*pi*5*x / 5.9)
    
    params: array of length 10, coefficients for each basis function.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Domain is [0.1, 6], so span is approximately 5.9
    span = 5.9
    freq_scale = 2.0 * np.pi / span
    
    # Precompute frequencies
    f1 = freq_scale * 1.0
    f2 = freq_scale * 2.0
    f3 = freq_scale * 3.0
    f4 = freq_scale * 4.0
    f5 = freq_scale * 5.0
    
    # Compute basis functions
    b0 = np.ones_like(x)
    b1 = np.cos(f1 * x)
    b2 = np.sin(f1 * x)
    b3 = np.cos(f2 * x)
    b4 = np.sin(f2 * x)
    b5 = np.cos(f3 * x)
    b6 = np.sin(f3 * x)
    b7 = np.cos(f4 * x)
    b8 = np.sin(f4 * x)
    b9 = np.cos(f5 * x)
    
    # Combine with parameters
    result = (params[0] * b0 +
              params[1] * b1 +
              params[2] * b2 +
              params[3] * b3 +
              params[4] * b4 +
              params[5] * b5 +
              params[6] * b6 +
              params[7] * b7 +
              params[8] * b8 +
              params[9] * b9)
    
    return result
