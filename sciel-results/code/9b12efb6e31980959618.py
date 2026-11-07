"""Symbolic regression: nonlinear oscillator acceleration with Anisotropic RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Precompute RBF centers and anisotropic widths
# Input ranges:
# x: [-0.2056, 0.1749]
# v: [-0.3391, 0.3624]

# Define 8 centers on a 2x4 grid in (x, v) space
# x has 2 centers, v has 4 centers
_x_centers = np.array([-0.15, 0.10])
_v_centers = np.array([-0.30, -0.10, 0.10, 0.30])

_xc, _vc = np.meshgrid(_x_centers, _v_centers, indexing='ij')
_centers = np.column_stack([_xc.ravel(), _vc.ravel()])  # shape (8, 2)

# Anisotropic bandwidths to match input scales
# x scale ~ 0.2, v scale ~ 0.35
# We want RBFs to cover the domain. 
# sigma_x = 0.15, sigma_v = 0.25
_sigma_x = 0.15
_sigma_v = 0.25

# Precompute inverse variances for efficiency
_inv_2_sig_x2 = 1.0 / (2.0 * _sigma_x * _sigma_x)
_inv_2_sig_v2 = 1.0 / (2.0 * _sigma_v * _sigma_v)


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Anisotropic RBF expansion model for nonlinear oscillator acceleration.
    
    Uses 8 Gaussian RBFs centered on a 2x4 grid in (x, v) space with anisotropic widths.
    Includes explicit linear terms for position (restoring) and velocity (damping).
    
    params[0:8] = RBF weights
    params[8]   = coefficient for linear x term
    params[9]   = coefficient for linear v term
    """
    p = params  # shape (10,)
    
    # Compute anisotropic squared distances
    # x shape (N,), v shape (N,)
    # centers shape (8, 2)
    
    dx = x[:, np.newaxis] - _centers[:, 0][np.newaxis, :]  # (N, 8)
    dv = v[:, np.newaxis] - _centers[:, 1][np.newaxis, :]  # (N, 8)
    
    # Anisotropic distance squared
    dist2 = (dx * dx) * _inv_2_sig_x2 + (dv * dv) * _inv_2_sig_v2  # (N, 8)
    
    # Gaussian RBF values
    rbf_vals = np.exp(-dist2)  # (N, 8)
    
    # Weighted sum of RBFs
    rbf_sum = rbf_vals @ p[:8]  # (N,)
    
    # Linear terms for restoring force and damping
    linear_x = p[8] * x
    linear_v = p[9] * v
    
    return rbf_sum + linear_x + linear_v
# EVOLVE-BLOCK-END
