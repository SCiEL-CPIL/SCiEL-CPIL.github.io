"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: evaluator kills after 150s wall-clock. Keep a safety margin.
_DEADLINE_SECONDS = 145.0
_START = time.time()


def _elapsed_ok() -> bool:
    return (time.time() - _START) < _DEADLINE_SECONDS


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using a Radial Basis Function (RBF) expansion.
    
    Instead of a global polynomial/trig basis, we use 10 Gaussian RBFs centered 
    at strategically chosen points in the (t, x, v) space. The 10 parameters 
    are the weights of these RBFs. This allows the model to capture local 
    nonlinear behaviors and complex phase-space structures that a global 
    polynomial might miss, while remaining smooth and differentiable for BFGS.
    
    Parameters:
    t, x, v: Input arrays (10000 rows)
    params: 10 weights for the RBFs
    """
    # Safety margin guard
    if not _elapsed_ok():
        return np.zeros(np.shape(t), dtype=float)

    # Ensure float arrays and consistent shape
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)

    p = np.asarray(params, dtype=float)
    # Defensive: if params is short, pad with zeros; if long, ignore extras.
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]))
    p = p[:10]

    # Define RBF centers in the (t, x, v) space.
    # t range: [30, 50] -> center at 40
    # x range: [-0.2, 0.17] -> centers at -0.1, 0.0, 0.1
    # v range: [-0.34, 0.36] -> centers at -0.2, 0.0, 0.2
    
    # We create 10 distinct centers to span the input space.
    # This is a fixed structural choice; the weights are fitted.
    centers = np.array([
        [30.0, -0.20, -0.30],  # Corner 1
        [50.0,  0.17,  0.36],  # Corner 2
        [40.0,  0.00,  0.00],  # Center
        [35.0, -0.10,  0.20],  # Quadrant 1
        [45.0,  0.10, -0.20],  # Quadrant 2
        [30.0,  0.10,  0.00],  # Edge 1
        [50.0, -0.10,  0.00],  # Edge 2
        [40.0, -0.20,  0.30],  # Edge 3
        [40.0,  0.17, -0.30],  # Edge 4
        [37.5,  0.05,  0.15]   # Intermediate
    ])

    # Input data matrix (N, 3)
    inputs = np.column_stack((t, x, v))
    
    # Calculate distances from each input to each center (N, 10)
    # dist[i, j] = || inputs[i] - centers[j] ||^2
    # Using broadcasting: (N, 1, 3) - (1, 10, 3) -> (N, 10, 3)
    diffs = inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diffs * diffs, axis=2)
    
    # RBF Gaussian: exp(-sigma * dist_sq)
    # We need to choose sigma. A fixed sigma is part of the structure.
    # The ranges are: t~20, x~0.4, v~0.7.
    # Normalizing inputs would help, but we can't modify inputs easily without extra params.
    # Let's use a relatively narrow sigma to allow local fitting.
    # If sigma is too small, it becomes a lookup table (overfitting risk).
    # If sigma is too large, it becomes a global polynomial (underfitting).
    # Given the ranges, a sigma around 0.5-2.0 might be appropriate for the normalized scale,
    # but here we are using raw units.
    # t diff can be up to 20. dist_sq up to 400. exp(-1*400) is 0.
    # x diff up to 0.4. dist_sq up to 0.16.
    # v diff up to 0.7. dist_sq up to 0.5.
    # The t dimension dominates the distance.
    # To make the RBFs sensitive to x and v as well, we should weight the dimensions or use a larger sigma.
    # Let's try sigma = 0.1. 
    # dist_sq for t=10 away is 100. exp(-10) ~ 4e-5.
    # dist_sq for x=0.2 away is 0.04. exp(-0.004) ~ 0.996.
    # This means the RBFs are very narrow in t but wide in x/v.
    # This might be good if the dynamics are periodic in t but depend on state.
    
    sigma = 0.1
    rbf_values = np.exp(-sigma * dist_sq)
    
    # The output is the weighted sum of the RBFs
    # out = rbf_values @ p
    out = np.sum(rbf_values * p, axis=1)
    
    # Guarantee finiteness and correct shape
    out = np.where(np.isfinite(out), out, 0.0)
    return np.asarray(out, dtype=float)
# EVOLVE-BLOCK-END
