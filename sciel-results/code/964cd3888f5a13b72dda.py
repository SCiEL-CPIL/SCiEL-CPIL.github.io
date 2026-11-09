"""Symbolic regression with RBF basis: localized Gaussians + global trend."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: linear combination of 10 basis functions of x.
    Strategy: localized Gaussian RBFs at distinct centers within [-3,3]
    combined with a global linear trend and one global smooth term.
    
    Basis:
      0: 1 (constant)
      1: x (linear trend)
      2: exp(-(x+2.5)^2 / 2s^2)  RBF centered at -2.5
      3: exp(-(x+1.5)^2 / 2s^2)  RBF centered at -1.5
      4: exp(-(x+0.5)^2 / 2s^2)  RBF centered at -0.5
      5: exp(-(x-0.5)^2 / 2s^2)  RBF centered at  0.5
      6: exp(-(x-1.5)^2 / 2s^2)  RBF centered at  1.5
      7: exp(-(x-2.5)^2 / 2s^2)  RBF centered at  2.5
      8: tanh(x) (global smooth nonlinearity)
      9: sin(x)  (global periodic component)
    """
    # Ensure x is 1D array
    x = np.asarray(x, dtype=np.float64)
    
    # For numerical stability, clip input
    x_clip = np.clip(x, -10, 10)
    
    # Global basis
    b0 = np.ones_like(x_clip)
    b1 = x_clip
    
    # Localized Gaussian RBFs with moderate width
    # sigma = 0.8 gives reasonable overlap for centers spaced 1.0 apart
    sigma2 = 0.64  # sigma = 0.8
    inv_2s2 = 1.0 / (2.0 * sigma2)
    
    centers = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    
    # Build RBF basis vectors
    b2 = np.exp(-inv_2s2 * (x_clip - centers[0]) ** 2)
    b3 = np.exp(-inv_2s2 * (x_clip - centers[1]) ** 2)
    b4 = np.exp(-inv_2s2 * (x_clip - centers[2]) ** 2)
    b5 = np.exp(-inv_2s2 * (x_clip - centers[3]) ** 2)
    b6 = np.exp(-inv_2s2 * (x_clip - centers[4]) ** 2)
    b7 = np.exp(-inv_2s2 * (x_clip - centers[5]) ** 2)
    
    # Global smooth terms
    b8 = np.tanh(x_clip)
    b9 = np.sin(x_clip)
    
    # Stack into matrix (N, 10)
    basis = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Linear combination
    y = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y
# EVOLVE-BLOCK-END
