import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: keep a safety margin before the 150s wall-clock limit
_DEADLINE = 150.0
_SAFETY_MARGIN = 15.0  # seconds to leave spare


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev tensor product (degree <= 3) + trigonometric terms.
    
    This combines the well-conditioned polynomial structure from the current/inspiration programs
    with the trigonometric flexibility from inspiration 1, which scored significantly better.
    
    Basis (10 terms):
      0: T0(x)T0(y) = 1
      1: T1(x)T0(y) = x
      2: T0(x)T1(y) = y
      3: T2(x)T0(y) = 2x^2 - 1
      4: T0(x)T2(y) = 2y^2 - 1
      5: T1(x)T1(y) = xy
      6: sin(pi * x)
      7: sin(pi * y)
      8: cos(pi * x)
      9: cos(pi * y)
    
    The first 6 terms span all bivariate polynomials up to degree 2 (with Chebyshev basis),
    and the last 4 terms capture periodic structure that pure polynomials cannot.
    """
    # Chebyshev polynomials of the first kind
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x**2 - 1.0
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y**2 - 1.0
    
    # Trigonometric terms
    sin_px = np.sin(np.pi * x)
    sin_py = np.sin(np.pi * y)
    cos_px = np.cos(np.pi * x)
    cos_py = np.cos(np.pi * y)
    
    # Tensor product Chebyshev terms (polynomial part)
    f = (params[0] * T0_x * T0_y +
         params[1] * T1_x * T0_y +
         params[2] * T0_x * T1_y +
         params[3] * T2_x * T0_y +
         params[4] * T0_x * T2_y +
         params[5] * T1_x * T1_y +
         params[6] * sin_px +
         params[7] * sin_py +
         params[8] * cos_px +
         params[9] * cos_py)
         
    return f
