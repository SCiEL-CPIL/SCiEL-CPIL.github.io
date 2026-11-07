# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate a linear combination of 10 basis functions at (x, y).
    
    Basis functions chosen to be well-conditioned for BFGS optimization
    starting from params = ones(10):
      0: constant
      1: x
      2: y
      3: x^2
      4: y^2
      5: xy
      6: sin(pi*x)
      7: sin(pi*y)
      8: x^3
      9: y^3
    """
    # Compute basis matrix: shape (n_points, 10)
    phi = np.column_stack([
        np.ones_like(x),
        x,
        y,
        x ** 2,
        y ** 2,
        x * y,
        np.sin(np.pi * x),
        np.sin(np.pi * y),
        x ** 3,
        y ** 3,
    ])
    
    # Linear combination: (n_points, 10) @ (10,) -> (n_points,)
    pred = phi @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
