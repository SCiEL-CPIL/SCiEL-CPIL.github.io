"""Symbolic regression using a diverse global basis of polynomial, trigonometric, rational, and exponential terms."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model using a diverse global basis.
    
    Basis (10 terms):
    b0 = 1
    b1 = x
    b2 = x**2
    b3 = x**3
    b4 = sin(x)
    b5 = cos(x)
    b6 = sin(2x)
    b7 = cos(2x)
    b8 = 1 / (1 + x)
    b9 = exp(-x)
    
    This basis is designed to be:
    1. Well-conditioned for BFGS optimization from an all-ones start.
    2. Flexible enough to capture polynomial trends, periodicity, rational decay, 
       and exponential decay.
    3. Globally defined over the input range [0.1, 6], avoiding the local 
       minima and conditioning issues associated with fixed-center RBFs.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = params
    
    # Polynomial basis
    b0 = 1.0
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    
    # Trigonometric basis (frequencies 1 and 2)
    b4 = np.sin(x)
    b5 = np.cos(x)
    b6 = np.sin(2.0 * x)
    b7 = np.cos(2.0 * x)
    
    # Rational basis
    b8 = 1.0 / (1.0 + x)
    
    # Exponential decay basis
    b9 = np.exp(-x)
    
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
