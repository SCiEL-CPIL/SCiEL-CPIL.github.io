import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation that combines a polynomial basis (degrees 0-7) and a periodic basis
    (sin/cos for frequencies 1 and 2) to model a wide range of functions.
    
    params:
      0: a0 (constant)
      1: a1 (x)
      2: a2 (x^2)
      3: a3 (x^3)
      4: a4 (x^4)
      5: a5 (x^5)
      6: a6 (x^6)
      7: a7 (x^7)
      8: b1 (sin(x))
      9: b2 (cos(x))
    """
    # Polynomial terms
    p0 = np.ones_like(x)
    p1 = x
    p2 = x**2
    p3 = x**3
    p4 = x**4
    p5 = x**5
    p6 = x**6
    p7 = x**7
    
    # Periodic terms
    s1 = np.sin(x)
    c1 = np.cos(x)
    
    # Combine
    result = (params[0] * p0 +
              params[1] * p1 +
              params[2] * p2 +
              params[3] * p3 +
              params[4] * p4 +
              params[5] * p5 +
              params[6] * p6 +
              params[7] * p7 +
              params[8] * s1 +
              params[9] * c1)
    
    return result
