# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 basis functions designed to span a rich function space
    on [-2, 2]. The basis includes:
      0: constant
      1: x
      2: x^2
      3: x^3
      4: x^4
      5: sin(x)
      6: cos(x)
      7: tanh(x)
      8: exp(-x^2)
      9: exp(-|x|)
    
    This mixed basis covers polynomials, periodic functions, saturating functions,
    and decaying functions, giving BFGS a good chance of fitting diverse targets.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    basis = np.empty((n, 10), dtype=np.float64)
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = x ** 2
    basis[:, 3] = x ** 3
    basis[:, 4] = x ** 4
    basis[:, 5] = np.sin(x)
    basis[:, 6] = np.cos(x)
    basis[:, 7] = np.tanh(x)
    basis[:, 8] = np.exp(-x ** 2)
    basis[:, 9] = np.exp(-np.abs(x))
    
    return basis @ params
# EVOLVE-BLOCK-END
