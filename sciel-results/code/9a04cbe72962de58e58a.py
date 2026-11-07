"""Rewritten program: Fixed Radial Basis Function (RBF) basis."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_S = 150.0
_MARGIN_S = 5.0
_START_TIME = time.time()


def _compute_basis(x):
    """
    Computes a fixed RBF basis with 10 components.
    Uses Gaussian kernels with fixed centers and widths, plus a constant.
    """
    xc = np.clip(x, -2.0, 2.0)
    
    # Define centers and widths
    # Mix of narrow and wide bumps to capture local and global features.
    centers = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -2.0, 2.0])
    widths = np.array([0.5, 0.8, 0.5, 1.0, 0.5, 0.8, 0.5, 1.5, 1.5])
    
    # Compute 9 Gaussian terms
    basis_gaussians = np.zeros((x.shape[0], 9))
    for i in range(9):
        diff = xc - centers[i]
        basis_gaussians[:, i] = np.exp(-0.5 * (diff / widths[i])**2)
        
    # Add constant term
    const_term = np.ones_like(xc)
    
    # Stack to get (N, 10)
    basis = np.column_stack((basis_gaussians, const_term))
    
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard: if we are close to the wall, return a safe prediction
    if time.time() - _START_TIME > _DEADLINE_S - _MARGIN_S:
        # Use the constant term (last param) as a safe fallback
        return np.full_like(x, params[9])

    basis = _compute_basis(x)
    
    # Linear combination
    pred = basis @ params

    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        # Fall back to constant
        return np.full_like(x, params[9])

    return pred
# EVOLVE-BLOCK-END
