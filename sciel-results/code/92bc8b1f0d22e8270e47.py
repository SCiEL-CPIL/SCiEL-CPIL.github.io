"""Symbolic regression with a rich, well-conditioned basis for BFGS fitting."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate an unknown function y = f(x) for x in [-2, 2] using a linear
    combination of 10 basis functions. The parameters are fit by BFGS starting
    from ones.
    
    The basis is chosen to be well-conditioned and to span a wide variety of
    functional forms:
      0: 1
      1: x
      2: x^2
      3: x^3
      4: sin(pi * x)
      5: cos(pi * x)
      6: sin(2 * pi * x)
      7: cos(2 * pi * x)
      8: tanh(x)
      9: x * tanh(x)
    """
    # Ensure x is 1-D
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Guard against non-finite params
    if not np.all(np.isfinite(p)):
        p = np.ones(10)
    
    n = x.size
    if n == 0:
        return np.empty(0)
    
    # Build basis matrix (n x 10)
    B = np.empty((n, 10), dtype=np.float64)
    B[:, 0] = 1.0
    B[:, 1] = x
    B[:, 2] = x * x
    B[:, 3] = x * x * x
    B[:, 4] = np.sin(np.pi * x)
    B[:, 5] = np.cos(np.pi * x)
    B[:, 6] = np.sin(2.0 * np.pi * x)
    B[:, 7] = np.cos(2.0 * np.pi * x)
    B[:, 8] = np.tanh(x)
    B[:, 9] = x * np.tanh(x)
    
    return B @ p
# EVOLVE-BLOCK-END
