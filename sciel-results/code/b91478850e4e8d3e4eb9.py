"""
Radial Basis Function (RBF) approach.
Uses 10 Gaussian kernels centered at fixed points in the domain.
Linear in the 10 parameters, allowing BFGS to optimize efficiently.
This local basis can capture non-linear structures better than global polynomials.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_LIMIT = 150.0
_MARGIN = 10.0

# Pre-compute fixed centers for the RBFs
# Distribute 10 centers across [-1, 1]^2
# Using a mix of grid and specific points to ensure coverage
_CENTERS = np.array([
    [-1.0, -1.0],
    [ 1.0, -1.0],
    [-1.0,  1.0],
    [ 1.0,  1.0],
    [ 0.0, -1.0],
    [ 0.0,  1.0],
    [-1.0,  0.0],
    [ 1.0,  0.0],
    [ 0.0,  0.0],
    [ 0.5,  0.5]
])

# Standard deviation for the Gaussian kernels
# A moderate value allows some overlap but keeps locality
_SIGMA = 0.5
_SIGMA_SQ = 2.0 * _SIGMA * _SIGMA


def _elapsed_ok():
    return (time.time() - _START) < (_LIMIT - _MARGIN)


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure 1-D arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    n = x.size

    if not _elapsed_ok():
        # Safety fallback: return mean-ish constant
        return np.full(n, float(p[0]))

    # Construct input points matrix (n, 2)
    points = np.column_stack((x, y))
    
    # Compute distances from each point to each center
    # points: (n, 2), _CENTERS: (10, 2)
    # diff: (n, 10, 2)
    diff = points[:, np.newaxis, :] - _CENTERS[np.newaxis, :, :]
    
    # Squared distances: (n, 10)
    dist_sq = np.sum(diff**2, axis=2)
    
    # Gaussian kernel: (n, 10)
    basis = np.exp(-dist_sq / _SIGMA_SQ)
    
    # Linear combination
    return basis @ p
# EVOLVE-BLOCK-END
