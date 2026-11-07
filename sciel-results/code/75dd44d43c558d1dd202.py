"""Program using a Kernel Ridge Regression strategy with Gaussian RBF kernels."""
import numpy as np
import time

_start_time = time.time()


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form symbolic regression candidate.
    Uses a linear combination of 10 Gaussian RBF kernels with fixed centers.
    The 10 params act as adaptive weights for local interpolation.
    
    This shifts from global functional basis (polynomials/trig) to local flexibility,
    enabling the model to capture arbitrary smooth functions by weighting local bumps.
    """
    # Scale input to [-1, 1] for better kernel behavior
    u = x / 3.0
    
    # Define 10 fixed centers spread across the domain [-1, 1]
    centers = np.linspace(-1.0, 1.0, 10)
    
    # Choose bandwidth (sigma) for RBF kernels
    # A moderate bandwidth allows local flexibility while maintaining smoothness
    sigma = 0.4
    
    # Compute RBF values: exp(-((u - center)^2) / (2 * sigma^2))
    # u has shape (n,), centers has shape (10,)
    # We want output shape (n, 10)
    diff = u[:, np.newaxis] - centers[np.newaxis, :]
    rbf = np.exp(-diff**2 / (2 * sigma**2))
    
    # Linear combination: y = rbf @ params
    y = rbf @ params
    
    return y
