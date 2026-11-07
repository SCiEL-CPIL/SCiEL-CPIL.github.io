# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression: recover unknown f(x, y, z) using a flexible basis.
    
    The evaluator fits `params` (length 10) via BFGS starting from ones.
    We design a basis that is:
      1. Well-conditioned (centered inputs)
      2. Flexible enough to capture common functional forms
      3. Uses exactly 10 parameters
    
    Basis design (10 terms):
      p0: 1 (intercept)
      p1: xc = x - 1.25
      p2: yc = y - 1.25
      p3: zc = z - 1.25
      p4: xc^2
      p5: yc^2
      p6: zc^2
      p7: xc*yc
      p8: xc*zc
      p9: yc*zc
    
    This is a complete quadratic in centered variables, which should handle
    most smooth functions well and gives the BFGS optimizer a well-conditioned
    problem.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Center inputs to improve conditioning
    xc = x - 1.25
    yc = y - 1.25
    zc = z - 1.25
    
    p = params
    
    # Quadratic basis in centered variables
    pred = (p[0]
            + p[1] * xc
            + p[2] * yc
            + p[3] * zc
            + p[4] * xc * xc
            + p[5] * yc * yc
            + p[6] * zc * zc
            + p[7] * xc * yc
            + p[8] * xc * zc
            + p[9] * yc * zc)
    
    return pred
# EVOLVE-BLOCK-END
