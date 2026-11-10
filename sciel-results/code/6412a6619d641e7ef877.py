"""Rewritten program for symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s limit

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a sum of 5 Gaussian basis functions.
    
    Parameters are organized as:
    p[0:5]  -> amplitudes of the 5 Gaussians
    p[5:10] -> means of the 5 Gaussians
    
    Widths are fixed at 1.0 to keep the optimization landscape smoother
    while still allowing localized, non-linear features to be captured.
    This radial basis function approach offers a different inductive bias
    than global polynomial or Chebyshev bases, enabling adaptation to
    sharp features, discontinuities, or multi-modal structures.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Amplitudes and means for 5 Gaussian basis functions
    amplitudes = p[0:5]
    means = p[5:10]
    widths = np.ones(5)
    
    # Compute the sum of Gaussians: sum_i amp_i * exp(-0.5 * ((x - mean_i) / width_i)^2)
    # Vectorized computation for efficiency
    result = np.zeros_like(x)
    for i in range(5):
        diff = x - means[i]
        result += amplitudes[i] * np.exp(-0.5 * (diff / widths[i])**2)
    
    return result

# EVOLVE-BLOCK-END
