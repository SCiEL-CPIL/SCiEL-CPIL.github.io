"""Piecewise cubic B-spline with fixed knots for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Cubic B-spline interpolation with 13 fixed knots (10 basis functions).
    params: 10 control values.
    """
    # 13 evenly spaced knots in [-2, 2]
    knots = np.linspace(-2.0, 2.0, 13)
    n_knots = len(knots)
    n_basis = n_knots - 3  # 10 basis functions
    
    # Clip x to knot range
    x_clip = np.clip(x, knots[0], knots[-1])
    N = x_clip.shape[0]
    
    # Compute cubic B-spline basis using Cox-de Boor recurrence
    # Order 1: M_1[i](x) = 1 if knots[i] <= x < knots[i+1]
    M = np.zeros((N, n_knots - 1), dtype=np.float64)
    for i in range(n_knots - 1):
        mask = (x_clip >= knots[i]) & (x_clip < knots[i + 1])
        M[mask, i] = 1.0
    # Handle right edge
    mask = (x_clip == knots[-1])
    M[mask, n_knots - 2] = 1.0
    
    # Recurrence to order 4 (cubic)
    for r in range(1, 4):
        new_M = np.zeros((N, n_knots - r), dtype=np.float64)
        for i in range(n_knots - r):
            d1 = knots[i + r] - knots[i]
            d2 = knots[i + r + 1] - knots[i + 1]
            if d1 > 1e-15:
                new_M[:, i] += (x_clip - knots[i]) / d1 * M[:, i]
            if d2 > 1e-15:
                new_M[:, i] += (knots[i + r + 1] - x_clip) / d2 * M[:, i + 1]
        M = new_M
    
    # M now has shape (N, n_basis) = (N, 10)
    # Ensure no NaN/Inf
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    
    return M @ params
# EVOLVE-BLOCK-END
