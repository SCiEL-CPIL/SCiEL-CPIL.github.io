"""Symbolic regression with 1 input(s) - Richer basis with adaptive optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a rich set of basis functions to recover an unknown function.
    
    Basis functions:
      0: 1                          (constant)
      1: x                          (linear)
      2: x^2                        (quadratic)
      3: x^3                        (cubic)
      4: x^4                        (quartic)
      5: tanh(x)                    (gentle saturation)
      6: tanh(2*x)                  (steeper saturation)
      7: exp(-x^2)                  (Gaussian bump)
      8: sin(x)                     (sinusoidal)
      9: cos(x)                     (cosinusoidal)
    
    The expression is a linear combination of these basis functions, making it
    easy for BFGS to optimize the 10 parameters.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial terms
    poly = p0 + p1 * x + p2 * x**2 + p3 * x**3 + p4 * x**4
    
    # Saturation terms
    sat1 = p5 * np.tanh(x)
    sat2 = p6 * np.tanh(2.0 * x)
    
    # Gaussian bump
    gauss = p7 * np.exp(-x**2)
    
    # Trigonometric terms
    trig_sin = p8 * np.sin(x)
    trig_cos = p9 * np.cos(x)
    
    return poly + sat1 + sat2 + gauss + trig_sin + trig_cos
# EVOLVE-BLOCK-END
