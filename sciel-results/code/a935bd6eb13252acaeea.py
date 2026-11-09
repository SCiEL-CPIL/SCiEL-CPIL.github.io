"""Symbolic regression candidate using a normalized Gaussian RBF basis."""
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses 10 Gaussian Radial Basis Functions with fixed centers and widths.
    The parameters are the weights for each RBF.
    Centers are chosen on a grid to cover the domain [-1, 1] x [-1, 1].
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    # Define 10 centers (2x5 grid)
    cx = np.array([-0.8, 0.8])
    cy = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])
    
    centers = np.array([(a, b) for a in cx for b in cy])
    
    # Fixed width
    sigma = 0.5
    
    # Compute basis values
    # X shape: (N, 10)
    basis = np.zeros((len(x), 10))
    for i, (cxi, cyi) in enumerate(centers):
        dist_sq = (x - cxi)**2 + (y - cyi)**2
        basis[:, i] = np.exp(-dist_sq / (2 * sigma**2))
        
    # Linear combination
    return basis @ params
