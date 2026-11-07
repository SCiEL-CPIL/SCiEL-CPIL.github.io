"""Symbolic regression for a nonlinear oscillator using RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

_START_TIME = time.time()
_DEADLINE = _START_TIME + 135  # 15s buffer before 150s kill

def _time_left() -> float:
    return _DEADLINE - time.time()

# Fixed Radial Basis Function parameters
# Input ranges: x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
# We define 10 centers in a grid pattern to cover the domain.
# The RBF basis functions are: exp(-||[x, v] - c_i||^2 / (2 * sigma^2))
# Since we have 10 parameters, we use 10 centers.

# Grid points for centers
# x centers: 5 points
x_centers = np.linspace(-0.55, 0.5, 5)
# v centers: 2 points (since we only have 10 params total, 5x2 grid)
v_centers = np.linspace(-0.15, 0.25, 2)

# Create meshgrid of centers
CX, CV = np.meshgrid(x_centers, v_centers)
# Flatten to get 10 centers
CENTERS_X = CX.flatten()
CENTERS_V = CV.flatten()

# Sigma (width) of the Gaussians. 
# The domain width is ~1.1 in x and ~0.45 in v.
# A sigma of 0.3 provides good overlap for smooth interpolation.
SIGMA = 0.3
SIGMA2 = 2 * SIGMA**2

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator using RBF basis.
    
    Args:
        x: position array
        v: velocity array  
        params: 10-element array of fitted constants (weights for each RBF)
    
    Returns:
        Array of acceleration predictions (one per row)
    """
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    # Compute distances from each point to each center
    # x shape: (N,), CENTERS_X shape: (10,)
    # We want a matrix of shape (N, 10)
    dx = x[:, np.newaxis] - CENTERS_X[np.newaxis, :]
    dv = v[:, np.newaxis] - CENTERS_V[np.newaxis, :]
    
    # Squared distances
    dist_sq = dx**2 + dv**2
    
    # RBF values: exp(-dist_sq / (2 * sigma^2))
    rbf_values = np.exp(-dist_sq / SIGMA2)
    
    # Linear combination of RBFs with weights from params
    # rbf_values shape: (N, 10), params shape: (10,)
    # Result shape: (N,)
    return rbf_values @ params

# The evaluator calls equation(x, v, params) directly after fitting.
# The fitting is done by the evaluator using scipy.optimize.minimize.
# We just need to provide the equation function.
# No _main or _fit_and_evaluate needed as the evaluator handles fitting.
# EVOLVE-BLOCK-END
