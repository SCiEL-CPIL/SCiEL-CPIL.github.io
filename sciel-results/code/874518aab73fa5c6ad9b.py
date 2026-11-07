"""
Strategy: Hybrid Multi-Scale Gaussian RBF Basis.

Uses 10 Gaussian RBF centers at fixed locations with two different widths:
- 5 narrower Gaussians (sigma=0.3) at the 5 interior points
- 5 wider Gaussians (sigma=0.7) at the 5 remaining points

This dual-scale approach allows the model to capture both sharp local features
(with narrow Gaussians) and broader global trends (with wide Gaussians) simultaneously.

The problem is linear in the 10 parameters, so BFGS will converge to the global
minimum of the training MSE. The fixed basis design provides inductive bias that
should generalize well to the test set.
"""
import numpy as np
import time

def _build_hybrid_rbf_basis(x: np.ndarray) -> np.ndarray:
    """
    Build the 10-column hybrid Gaussian RBF basis matrix with two scales.
    
    Args:
        x: Input array of shape (N,).
        
    Returns:
        phi: Basis matrix of shape (N, 10).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    
    # 10 centers uniformly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Assign two different sigmas:
    # Even-indexed centers (0, 2, 4, 6, 8) get narrow sigma = 0.3
    # Odd-indexed centers (1, 3, 5, 7, 9) get wide sigma = 0.7
    sigmas = np.array([0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7])
    
    # Compute basis values
    # phi[i, j] = exp(-(x[i] - centers[j])^2 / (2 * sigmas[j]^2))
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    # Reshape sigmas for broadcasting: (1, 10)
    sigma2 = 2.0 * sigmas[np.newaxis, :] ** 2
    phi = np.exp(-(diff ** 2) / sigma2)
    
    return phi

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Compute the predicted y values using the hybrid Gaussian RBF basis.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 coefficients.
        
    Returns:
        predictions: Array of shape (N,).
    """
    x = np.asarray(x, dtype=float).ravel()
    params = np.asarray(params, dtype=float).ravel()
    
    # Ensure params is length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
        
    # Build basis and compute linear combination
    phi = _build_hybrid_rbf_basis(x)
    result = phi @ params
    
    # Safety check for numerical stability
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
