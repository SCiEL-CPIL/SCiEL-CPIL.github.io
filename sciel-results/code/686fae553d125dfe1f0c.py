"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: cubic polynomial + 2 tunable sigmoid units (exactly 10 params).
    y(x) = a0 + a1*x + a2*x^2 + a3*x^3 + c1*sigmoid(w1*x+b1) + c2*sigmoid(w2*x+b2)
    
    Params (10 total):
      a0, a1, a2, a3 : polynomial coefficients
      w1, b1, c1     : first sigmoid unit
      w2, b2, c2     : second sigmoid unit
    """
    a0, a1, a2, a3, w1, b1, c1, w2, b2, c2 = params
    
    # Cubic polynomial part (Horner's method)
    poly = a0 + x * (a1 + x * (a2 + x * a3))
    
    # Stable sigmoid to avoid overflow
    z1 = w1 * x + b1
    z2 = w2 * x + b2
    
    sig1 = np.where(z1 >= 0, 1.0 / (1.0 + np.exp(-z1)), np.exp(z1) / (1.0 + np.exp(z1)))
    sig2 = np.where(z2 >= 0, 1.0 / (1.0 + np.exp(-z2)), np.exp(z2) / (1.0 + np.exp(z2)))
    
    result = poly + c1 * sig1 + c2 * sig2
    
    return result
# EVOLVE-BLOCK-END
