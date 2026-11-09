"""Symbolic regression with 2 inputs: Radial Basis Function (RBF) basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 145.0

# Pre-compute fixed centers for the RBF basis
# A grid of centers covering [-1, 1] x [-1, 1]
# We need exactly 10 features to match the 10 params.
# Using a 3x3 grid gives 9 centers. We add one more at the center or corners.
# Let's use a 3x3 grid for the first 9, and the 10th can be a constant or another specific point.
# Actually, a constant term (offset) is usually very useful.
# Let's define 9 Gaussian centers and 1 constant term.

# Centers for 3x3 grid
xs = np.array([-0.8, 0.0, 0.8])
ys = np.array([-0.8, 0.0, 0.8])
X, Y = np.meshgrid(xs, ys, indexing='ij')
centers = np.column_stack((X.ravel(), Y.ravel()))

# Width (sigma) of the Gaussian kernels.
# If sigma is too small, the basis is very localized (hard to fit with BFGS from init).
# If sigma is too large, it behaves like a global polynomial (less expressive).
# A moderate value around 0.5 - 1.0 is typical for this domain.
# Let's try sigma = 0.6.
SIGMA = 0.6
SIGMA_SQ = SIGMA * SIGMA * 2.0  # Denominator in exponent: exp(-d^2 / (2*sigma^2))

def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return a (n, 10) feature matrix using Gaussian RBFs and a constant.
    """
    n = len(x)
    
    # Compute squared distances from each point (x_i, y_i) to each center (c_jx, c_jy)
    # x shape: (n, 1)
    # centers shape: (9, 2)
    # dist_sq shape: (n, 9)
    
    # Efficient distance calculation
    # d^2 = (x - cx)^2 + (y - cy)^2
    # = x^2 - 2*x*cx + cx^2 + y^2 - 2*y*cy + cy^2
    
    x2 = x * x
    y2 = y * y
    
    # centers: (9, 2)
    cx = centers[:, 0]  # (9,)
    cy = centers[:, 1]  # (9,)
    cx2 = cx * cx
    cy2 = cy * cy
    c2 = cx2 + cy2  # (9,)
    
    # x @ cx.T -> (n, 9)
    # y @ cy.T -> (n, 9)
    dist_sq = x2[:, np.newaxis] + c2[np.newaxis, :] + y2[:, np.newaxis] - 2.0 * (x @ cx[np.newaxis, :] + y @ cy[np.newaxis, :])
    
    # Clip to avoid negative values due to floating point errors
    dist_sq = np.maximum(dist_sq, 0.0)
    
    # Gaussian kernels
    # phi_j = exp(-dist_sq / (2 * sigma^2))
    kernels = np.exp(-dist_sq / SIGMA_SQ)
    
    # Add constant term as the 10th feature
    const = np.ones((n, 1))
    
    F = np.hstack((kernels, const))
    return F

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Compute predictions as linear combination of 10 RBF features.
    """
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Guard against excessive runtime
    if time.time() > _DEADLINE:
        F = _features(x, y)
        pred = F @ params
        if not np.all(np.isfinite(pred)):
            pred = np.where(np.isfinite(pred), pred, 0.0)
        return pred

    F = _features(x, y)
    pred = F @ params

    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        mask = np.isfinite(pred)
        pred = np.where(mask, pred, 0.0)

    return pred
# EVOLVE-BLOCK-END
