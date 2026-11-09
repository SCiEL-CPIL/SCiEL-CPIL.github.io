"""
Strategy: Multi-Scale Gaussian Mixture Basis with Adaptive Widths.

The current program uses a uniform grid of 9 RBFs with a single fixed width.
This new strategy uses a "Gaussian Mixture" approach where the 10 basis
functions are divided into three groups with different widths:
1. 3 "Wide" Gaussians (sigma=2.5) to capture global trends.
2. 4 "Medium" Gaussians (sigma=1.0) to capture medium-scale features.
3. 3 "Narrow" Gaussians (sigma=0.5) to capture local details.

The centers are also distributed to ensure coverage:
- Wide: -3, 0, 3
- Medium: -2.25, -0.75, 0.75, 2.25
- Narrow: -1.5, 0, 1.5

This multi-scale representation is structurally different from a uniform RBF grid
as it explicitly separates the approximation of different frequency components,
which can help the optimizer (BFGS) converge more robustly by reducing
inter-parameter correlation (ill-conditioning) often found in uniform RBFs.
"""
# EVOLVE-BLOCK-START
import numpy as np

# Pre-compute centers and sigmas for the multi-scale basis
# Total 10 parameters.

# Group 1: Wide (3 centers)
_centers_wide = np.array([-3.0, 0.0, 3.0])
_sigmas_wide = np.array([2.5, 2.5, 2.5])

# Group 2: Medium (4 centers)
_centers_med = np.array([-2.25, -0.75, 0.75, 2.25])
_sigmas_med = np.array([1.0, 1.0, 1.0, 1.0])

# Group 3: Narrow (3 centers)
_centers_narrow = np.array([-1.5, 0.0, 1.5])
_sigmas_narrow = np.array([0.5, 0.5, 0.5])

# Combine into single arrays for efficient computation
_centers = np.concatenate([_centers_wide, _centers_med, _centers_narrow])
_sigmas = np.concatenate([_sigmas_wide, _sigmas_med, _sigmas_narrow])

# Pre-compute 1/(2*sigma^2) for each basis function
_inv_2_sigma2 = 1.0 / (2.0 * _sigmas**2)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a multi-scale Gaussian basis.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters (weights for each Gaussian).
    
    Returns:
        Array of predictions of shape (N,).
    """
    x = np.asarray(x).ravel()
    
    # x shape: (N,)
    # centers shape: (10,)
    
    # Compute differences: (N, 10)
    diff = x[:, None] - _centers[None, :]
    
    # Compute Gaussian values: exp(-diff^2 * inv_2_sigma2)
    # _inv_2_sigma2 shape: (10,)
    rbf_vals = np.exp(-diff**2 * _inv_2_sigma2[None, :])
    
    # rbf_vals shape: (N, 10)
    # params shape: (10,)
    
    # Linear combination
    y_pred = rbf_vals.dot(params)
    
    return y_pred

# EVOLVE-BLOCK-END
