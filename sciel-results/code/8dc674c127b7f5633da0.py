"""Symbolic regression candidate: rich polynomial + radial basis features."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model with 10 learnable coefficients.
    
    Features (10 total):
      1. 1
      2. x
      3. y
      4. x^2
      5. y^2
      6. x*y
      7. x^3
      8. y^3
      9. x^2*y
      10. x*y^2
    
    This is a degree-3 polynomial in two variables, which can exactly recover
    any cubic polynomial and approximate many smooth functions well on [-1,1]^2.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    p = params[:10]
    
    f1 = np.ones_like(x)
    f2 = x
    f3 = y
    f4 = x * x
    f5 = y * y
    f6 = x * y
    f7 = x * x * x
    f8 = y * y * y
    f9 = x * x * y
    f10 = x * y * y
    
    return (p[0] * f1 + p[1] * f2 + p[2] * f3 + p[3] * f4 + p[4] * f5 +
            p[5] * f6 + p[6] * f7 + p[7] * f8 + p[8] * f9 + p[9] * f10)
# EVOLVE-BLOCK-END
