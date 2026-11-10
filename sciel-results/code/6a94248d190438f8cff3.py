"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich linear-in-params basis with 10 diverse functions to approximate a hidden f(x).
    
    Basis functions span common smooth function families on [-2, 2]:
    - Polynomials (degree 0-4)
    - Trigonometric (sin(x), cos(x))
    - Gaussian (exp(-x^2))
    - Hyperbolic tangent (tanh(x))
    - Rational (x/(1+x^2))
    - Second harmonic sin(2x)
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Build basis functions
    b0 = np.ones_like(x)          # constant
    b1 = x                         # linear
    b2 = x**2                      # quadratic
    b3 = x**3                      # cubic
    b4 = x**4                      # quartic
    b5 = np.sin(x)                 # sine
    b6 = np.cos(x)                 # cosine
    b7 = np.exp(-x**2)             # Gaussian
    b8 = np.tanh(x)                # tanh
    b9 = x / (1.0 + x**2)          # rational
    
    # Linear combination
    return (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 +
            p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9)
# EVOLVE-BLOCK-END
