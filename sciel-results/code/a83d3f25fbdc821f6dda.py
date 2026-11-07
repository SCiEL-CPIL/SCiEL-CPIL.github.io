"""Symbolic regression: 10-parameter basis with rational and interaction terms."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter basis for 2D function recovery.
    
    Features:
      0: 1                         (constant)
      1: x                         (linear x)
      2: y                         (linear y)
      3: x^2                       (quadratic x)
      4: y^2                       (quadratic y)
      5: x*y                       (bilinear interaction)
      6: 1/(1+x^2)                 (rational in x)
      7: 1/(1+y^2)                 (rational in y)
      8: 1/(1+x^2+y^2)             (rational in both)
      9: x^2*y^2                   (quartic interaction)
    """
    p = np.asarray(params, dtype=float)
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    x2 = x * x
    y2 = y * y
    xy = x * y
    x2y2 = x2 * y2
    
    # Rational terms with bounded denominators
    r1 = 1.0 / (1.0 + x2)
    r2 = 1.0 / (1.0 + y2)
    r3 = 1.0 / (1.0 + x2 + y2)
    
    pred = (
        p[0] +
        p[1] * x +
        p[2] * y +
        p[3] * x2 +
        p[4] * y2 +
        p[5] * xy +
        p[6] * r1 +
        p[7] * r2 +
        p[8] * r3 +
        p[9] * x2y2
    )
    
    return pred
# EVOLVE-BLOCK-END
