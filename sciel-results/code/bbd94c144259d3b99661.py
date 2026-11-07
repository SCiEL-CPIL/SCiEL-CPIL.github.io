"""Rewritten program using a richer, well-conditioned basis for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict using a combination of 10 basis functions.
    The basis includes:
      1. Constant
      2. x
      3. x^2
      4. x^3
      5. x^4
      6. sin(x)
      7. cos(x)
      8. exp(x)
      9. tanh(x)
      10. |x|
    
    This gives the optimizer flexibility to fit a wide variety of smooth functions.
    """
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Compute basis functions
    basis = np.empty((10, len(x)), dtype=np.float64)
    basis[0] = 1.0
    basis[1] = x
    basis[2] = x ** 2
    basis[3] = x ** 3
    basis[4] = x ** 4
    basis[5] = np.sin(x)
    basis[6] = np.cos(x)
    basis[7] = np.exp(x)
    basis[8] = np.tanh(x)
    basis[9] = np.abs(x)
    
    # Return linear combination
    return basis @ params
# EVOLVE-BLOCK-END
