"""Rewritten program for symbolic regression with 1 input using RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Gaussian RBF basis expansion.
    
    We use 10 Gaussian radial basis functions centered at evenly spaced points
    in [0.1, 6] with a fixed width. The 10 parameters act as output weights.
    
    This linear-in-parameters model allows BFGS to efficiently find the best
    linear combination, fitting arbitrary smooth non-linear shapes without
    conditioning issues of high-degree polynomials.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # 10 evenly spaced centers in [0.1, 6]
    centers = np.linspace(0.1, 6.0, 10)
    
    # Fixed width for the Gaussian RBFs
    # The spacing between centers is (6.0 - 0.1) / 9 ≈ 0.656
    # A width of about half the spacing gives good local coverage
    width = 0.5
    
    # Compute RBF values: shape (n_points, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-0.5 * (diff / width) ** 2)
    
    # Linear combination: shape (n_points,)
    return rbf_values @ params
# EVOLVE-BLOCK-END
