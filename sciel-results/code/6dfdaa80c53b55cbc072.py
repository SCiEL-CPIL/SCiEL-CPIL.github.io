"""Symbolic regression with 1 input; fresh design with rich basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich basis model with 10 parameters.
    Basis functions:
    0: 1
    1: x
    2: x^2
    3: x^3
    4: x^4
    5: sin(x)
    6: cos(x)
    7: tanh(x)
    8: exp(-x^2/2)
    9: |x|
    """
    p = params
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    term0 = np.ones_like(x)
    term1 = x
    term2 = x2
    term3 = x3
    term4 = x4
    term5 = np.sin(x)
    term6 = np.cos(x)
    term7 = np.tanh(x)
    term8 = np.exp(-0.5 * x2)
    term9 = np.abs(x)
    
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
