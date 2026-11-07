"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters model with a rich, well-conditioned basis.
    BFGS from all-ones will converge to the global optimum since the
    model is linear in params.
    
    Basis functions (all bounded on [-2, 2]):
      0: 1
      1: x
      2: x^2
      3: x^3
      4: x^4
      5: sin(pi*x)
      6: cos(pi*x)
      7: sin(2*pi*x)
      8: cos(2*pi*x)
      9: exp(-x^2)
    """
    p = params
    pi = np.pi
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    y = (p[0] * 1.0 +
         p[1] * x +
         p[2] * x2 +
         p[3] * x3 +
         p[4] * x4 +
         p[5] * np.sin(pi * x) +
         p[6] * np.cos(pi * x) +
         p[7] * np.sin(2.0 * pi * x) +
         p[8] * np.cos(2.0 * pi * x) +
         p[9] * np.exp(-x2))
    
    return y
# EVOLVE-BLOCK-END
