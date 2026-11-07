"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Use a linear combination of 10 localized Gaussian basis functions
# centered at evenly spaced points across [-3, 3], with fixed widths.
# This forms a piecewise-smooth universal approximator that avoids the
# ill-conditioning of global polynomials and provides better local flexibility
# for BFGS convergence.

# Precompute centers and widths for the Gaussian basis functions
# 10 centers evenly spaced in [-3, 3]
_GAUSSIAN_CENTERS = np.linspace(-3.0, 3.0, 10)
# Width: chosen so that adjacent Gaussians have moderate overlap
# Distance between centers = 6.0 / 9 = 0.667
# Width ~ 0.5 gives good local coverage without excessive overlap
_GAUSSIAN_WIDTH = 0.5

def _build_gaussian_features(x: np.ndarray) -> np.ndarray:
    """
    Build a feature matrix of shape (n, 10) with localized Gaussian basis functions.
    Centers are evenly spaced across [-3, 3].
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    
    # Compute (x - center)^2 for all centers at once
    # x has shape (n,), centers has shape (10,)
    # diff has shape (n, 10)
    diff = x[:, np.newaxis] - _GAUSSIAN_CENTERS[np.newaxis, :]
    
    # Gaussian: exp(-0.5 * (diff / width)^2)
    F = np.exp(-0.5 * (diff / _GAUSSIAN_WIDTH) ** 2)
    
    return F


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form linear combination of 10 localized Gaussian basis functions.
    params: array of length 10, coefficients for each Gaussian.
    Returns: array of predictions, one per point.
    """
    F = _build_gaussian_features(x)
    return F @ params
# EVOLVE-BLOCK-END
