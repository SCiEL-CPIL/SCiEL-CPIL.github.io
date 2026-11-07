"""Symbolic regression with 1 input; adaptive basis with multiple frequency components."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive basis model with 10 parameters.
    
    Uses a combination of polynomial, trigonometric (multiple frequencies),
    and smooth nonlinear basis functions to span a wide class of functions.
    
    Basis functions:
    0: 1
    1: x
    2: x^2
    3: x^3
    4: sin(x)
    5: cos(x)
    6: sin(2x)
    7: cos(2x)
    8: tanh(x)
    9: exp(-x^2/2)
    """
    p = params
    
    x2 = x * x
    x3 = x2 * x
    
    term0 = np.ones_like(x)
    term1 = x
    term2 = x2
    term3 = x3
    term4 = np.sin(x)
    term5 = np.cos(x)
    term6 = np.sin(2 * x)
    term7 = np.cos(2 * x)
    term8 = np.tanh(x)
    term9 = np.exp(-0.5 * x2)
    
    y = (p[0] * term0 +
         p[1] * term1 +
         p[2] * term2 +
         p[3] * term3 +
         p[4] * term4 +
         p[5] * term5 +
         p[6] * term6 +
         p[7] * term7 +
         p[8] * term8 +
         p[9] * term9)
    
    return y
# EVOLVE-BLOCK-END
