"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF (Radial Basis Function) expansion for nonlinear oscillator acceleration.
    
    Uses 10 Gaussian RBFs with fixed centers spread across the input space.
    Each RBF has a width chosen to cover a reasonable neighborhood.
    The 10 parameters are the weights for each RBF.
    
    Input ranges:
      t in [30, 50]  -> centered at 40, half-range 10
      x in [-0.2056, 0.1749] -> centered at ~-0.015, half-range ~0.19
      v in [-0.3391, 0.3624] -> centered at ~0.01, half-range ~0.35
    
    We place centers on a coarse grid in 3D and use 10 of them.
    Widths are set so that each RBF has meaningful support.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalize inputs to roughly [-1, 1] for better conditioning
    tn = (t - 40.0) / 10.0       # roughly [-1, 1]
    xn = (x - (-0.015)) / 0.19   # roughly [-1, 1]
    vn = (v - 0.01) / 0.35       # roughly [-1, 1]

    # Define 10 fixed centers in normalized space
    # Spread them across the 3D input space
    centers = np.array([
        [-0.8, -0.5, -0.5],
        [-0.8,  0.5,  0.5],
        [ 0.0, -0.8,  0.0],
        [ 0.0,  0.0, -0.8],
        [ 0.0,  0.8,  0.8],
        [ 0.8, -0.5,  0.5],
        [ 0.8,  0.5, -0.5],
        [-0.4,  0.0,  0.0],
        [ 0.4,  0.0,  0.0],
        [ 0.0,  0.0,  0.0],
    ])

    # Widths (inverse of sigma) for each RBF
    # Larger width = narrower RBF
    widths = np.array([2.5, 2.5, 3.0, 3.0, 2.5, 2.5, 2.5, 3.5, 3.5, 4.0])

    # Compute RBF values
    # Input shape: (N, 3)
    inputs_norm = np.stack([tn, xn, vn], axis=-1)  # (N, 3)

    # For each center, compute squared distance
    # (N, 3) - (3,) -> (N, 3) -> sum over axis 1 -> (N,)
    pred = np.zeros_like(tn, dtype=np.float64)

    for i in range(10):
        diff = inputs_norm - centers[i]
        sq_dist = np.sum(diff * diff, axis=-1)
        rbf_val = np.exp(-widths[i] * sq_dist)
        pred += params[i] * rbf_val

    return pred
# EVOLVE-BLOCK-END
