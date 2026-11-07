"""Symbolic regression with 1 input(s) - Combined polynomial, sigmoid, and oscillatory basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Linear combination of 10 basis functions combining polynomial terms
    from the current program with flexible sigmoid units from inspiration 2,
    plus Gaussian and oscillatory components.
    
    Basis functions:
      0: 1                          (constant)
      1: x                          (linear)
      2: x^2                        (quadratic)
      3: x^3                        (cubic - from inspiration 1)
      4: sigmoid(2*x)               (gentle sigmoid centered at 0)
      5: sigmoid(3*(x-1))           (sigmoid centered at 1, steeper)
      6: sigmoid(3*(x+1))           (sigmoid centered at -1, steeper)
      7: exp(-x^2)                  (Gaussian bump at center)
      8: sin(x)                     (low-frequency oscillation)
      9: sin(2*x)                   (second harmonic)
    
    This combines the polynomial expressiveness of the current program with
    the localized non-linear features captured by sigmoid units, while keeping
    oscillatory terms for potential periodic components.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = params
    
    # Clip x to avoid numerical issues
    x_c = np.clip(x, -50, 50)
    
    # Polynomial basis
    b0 = np.ones_like(x_c)
    b1 = x_c
    b2 = x_c ** 2
    b3 = x_c ** 3
    
    # Flexible sigmoid units with fixed centers and slopes
    def sigmoid(z):
        # Numerically stable sigmoid
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    # Sigmoid unit 1: center=0, slope=2 -> sigmoid(2*x)
    b4 = sigmoid(2.0 * x_c)
    
    # Sigmoid unit 2: center=1, slope=3 -> sigmoid(3*(x-1))
    b5 = sigmoid(3.0 * (x_c - 1.0))
    
    # Sigmoid unit 3: center=-1, slope=3 -> sigmoid(3*(x+1))
    b6 = sigmoid(3.0 * (x_c + 1.0))
    
    # Gaussian bump
    exp_arg = -x_c ** 2
    exp_arg = np.maximum(exp_arg, -100)
    b7 = np.exp(exp_arg)
    
    # Sinusoidal terms
    b8 = np.sin(x_c)
    b9 = np.sin(2.0 * x_c)
    
    # Linear combination
    y = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
         p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
         p[8] * b8 + p[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
