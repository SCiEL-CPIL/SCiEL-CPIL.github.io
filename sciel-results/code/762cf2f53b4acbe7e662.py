"""
Strategy: A trigonometric harmonic basis with non-integer frequency multipliers.

Uses a linear baseline plus sine/cosine terms with fixed non-integer
frequencies chosen to capture variation over the input domain [0.5, 2].
Cross-interaction terms with mixed frequencies help model complex
dependencies between variables.

Model:
  f(x, y, z) = c0 + c1*x + c2*y + c3*z
               + w4 * sin(3.0*x + 2.0*y)
               + w5 * cos(2.5*y + 1.5*z)
               + w6 * sin(2.0*z + 3.0*x)
               + w7 * sin(4.0*x)
               + w8 * cos(3.5*y)
               + w9 * sin(3.0*z)

Parameters (10 total):
  0: c0       (intercept)
  1: c1       (x coefficient)
  2: c2       (y coefficient)
  3: c3       (z coefficient)
  4: w4       (weight for sin(3x+2y))
  5: w5       (weight for cos(2.5y+1.5z))
  6: w6       (weight for sin(2z+3x))
  7: w7       (weight for sin(4x))
  8: w8       (weight for cos(3.5y))
  9: w9       (weight for sin(3z))
"""
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns predictions using a linear baseline plus trigonometric harmonics
    with non-integer frequency multipliers.
    """
    c0, c1, c2, c3 = params[0], params[1], params[2], params[3]
    w4, w5, w6, w7, w8, w9 = params[4], params[5], params[6], params[7], params[8], params[9]
    
    # Linear baseline
    linear = c0 + c1 * x + c2 * y + c3 * z
    
    # Trigonometric terms with non-integer frequencies
    # Cross terms (interactions)
    term1 = w4 * np.sin(3.0 * x + 2.0 * y)
    term2 = w5 * np.cos(2.5 * y + 1.5 * z)
    term3 = w6 * np.sin(2.0 * z + 3.0 * x)
    
    # Individual higher-frequency terms
    term4 = w7 * np.sin(4.0 * x)
    term5 = w8 * np.cos(3.5 * y)
    term6 = w9 * np.sin(3.0 * z)
    
    return linear + term1 + term2 + term3 + term4 + term5 + term6
