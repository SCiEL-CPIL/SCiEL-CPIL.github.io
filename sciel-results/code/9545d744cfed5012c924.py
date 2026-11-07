"""Symbolic regression: 10-parameter RBF basis for 2D function recovery."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter Radial Basis Function (RBF) basis on [-1,1]^2.
    
    The first parameter is a global constant (bias). The remaining 9 parameters
    weight Gaussian RBFs centered at 9 fixed locations on a 3x3 grid.
    
    Centers:
      (-1, -1), (-1, 0), (-1, 1)
      ( 0, -1), ( 0, 0), ( 0, 1)
      ( 1, -1), ( 1, 0), ( 1, 1)
    
    Each RBF has a fixed width (sigma) chosen to provide overlapping coverage.
    This non-polynomial basis can capture smooth non-linear structures more
    robustly than high-degree polynomials, and BFGS can fit the weights well.
    """
    p = np.asarray(params, dtype=float)
    
    # 9 fixed centers on a 3x3 grid
    centers = [
        (-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0),
        ( 0.0, -1.0), ( 0.0, 0.0), ( 0.0, 1.0),
        ( 1.0, -1.0), ( 1.0, 0.0), ( 1.0, 1.0),
    ]
    
    # Fixed width for all RBFs. Chosen so that neighboring centers have
    # significant overlap but the basis is not too degenerate.
    sigma = 0.6
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)
    
    # Precompute the 9 RBF values
    rbf_vals = np.empty((len(centers), len(x)), dtype=float)
    for i, (cx, cy) in enumerate(centers):
        dx = x - cx
        dy = y - cy
        r2 = dx * dx + dy * dy
        rbf_vals[i] = np.exp(-inv_2s2 * r2)
    
    # Prediction: constant + weighted sum of RBFs
    pred = np.full(len(x), p[0], dtype=float)
    for i in range(9):
        pred += p[i + 1] * rbf_vals[i]
    
    return pred
# EVOLVE-BLOCK-END
