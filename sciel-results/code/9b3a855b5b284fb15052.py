"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Softplus basis function for symbolic regression.
    
    Uses 10 fixed softplus basis functions with centers and slopes on a
    symmetric grid within [-2, 2]. Only the weights are free parameters,
    making this a linear least-squares problem that BFGS can solve globally.
    """
    # Fixed centers and slopes for the softplus basis
    # We use 5 positive-slope and 5 negative-slope basis functions
    # Centers spread symmetrically across [-2, 2]
    
    # Centers: symmetric grid
    centers = np.array([-1.6, -0.8, 0.0, 0.8, 1.6, -1.6, -0.8, 0.0, 0.8, 1.6])
    # Slopes: first 5 positive, last 5 negative (for symmetry and flexibility)
    slopes = np.array([3.0, 3.0, 3.0, 3.0, 3.0, -3.0, -3.0, -3.0, -3.0, -3.0])
    
    # Compute softplus basis: softplus(s * (x - c)) = log(1 + exp(s * (x - c)))
    # To avoid overflow, use the stable formulation:
    # softplus(z) = max(z, 0) + log(1 + exp(-|z|))
    
    z = slopes[None, :] * (x[:, None] - centers[None, :])  # shape (n, 10)
    
    # Stable softplus
    softplus_vals = np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))
    
    # Linear combination with weights
    return params @ softplus_vals.T
# EVOLVE-BLOCK-END
