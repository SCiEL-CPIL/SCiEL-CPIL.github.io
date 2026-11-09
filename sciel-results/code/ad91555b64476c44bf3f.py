# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs (x, y) and 10 free parameters.
    Uses a hybrid basis: global polynomials up to degree 2, plus localized
    Gaussian bumps to capture local structure, plus a few trigonometric terms.
    
    The evaluator fits params (length 10) via BFGS starting from all ones.
    We design the basis so that BFGS from ones converges to a good fit.
    
    Basis functions (10 total, each multiplied by a param):
      0: 1 (constant)
      1: x
      2: y
      3: x^2
      4: y^2
      5: x*y
      6: sin(pi*x)
      7: sin(pi*y)
      8: exp(-4*(x^2 + y^2))   -- central Gaussian bump
      9: exp(-4*((x-0.5)^2 + (y-0.5)^2))  -- off-center bump
    
    This gives global quadratic trend + periodic + local structure.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Ensure p has at least 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.ones(10 - p.size)])
    
    # Basis
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * x
    b4 = y * y
    b5 = x * y
    b6 = np.sin(np.pi * x)
    b7 = np.sin(np.pi * y)
    
    r2 = x * x + y * y
    b8 = np.exp(-4.0 * r2)
    
    dx = x - 0.5
    dy = y - 0.5
    b9 = np.exp(-4.0 * (dx * dx + dy * dy))
    
    # Combine
    pred = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
            p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
            p[8] * b8 + p[9] * b9)
    
    # Guard against non-finite
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
# EVOLVE-BLOCK-END
