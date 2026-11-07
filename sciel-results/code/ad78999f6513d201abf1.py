"""
Symbolic regression: recover an unknown function f(x) with 1 input from [-3, 3].
Strategy: Linear combination of 10 localized Gaussian basis functions (RBFs).
Instead of a nonlinear model where centers/widths are parameters (which creates a
non-convex, multimodal landscape for BFGS), we use a FIXED linear basis of 10
Gaussians centered at evenly spaced points in [-3, 3] with fixed widths.
This makes the model linear in the 10 parameters, so BFGS converges quickly to the
global least-squares solution. The fixed centers provide local flexibility, and the
linear combination can approximate smooth functions well.

Model:
f(x) = sum_{i=0}^{9} p_i * exp(-(x - c_i)^2 / (2 * w^2))
where c_i are fixed centers evenly spaced in [-3, 3], and w is a fixed width.

Parameters (length 10):
p[0..9]: coefficients for each Gaussian.

This is a linear model, so BFGS should converge reliably.
"""

import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear RBF model with 10 fixed Gaussian basis functions.
    
    Params:
    p[0..9]: coefficients for 10 Gaussians centered at evenly spaced points in [-3, 3].
    
    Model:
    y = sum_{i=0}^{9} p_i * exp(-(x - c_i)^2 / (2 * w^2))
    """
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    orig_shape = x.shape
    x_flat = x.ravel()
    
    # Fixed centers: 10 evenly spaced points in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Fixed width: chosen so that Gaussians have some overlap but are localized.
    # Distance between adjacent centers is 6/9 = 0.667.
    # A width of ~0.5-0.8 gives good overlap. Let's use 0.7.
    w = 0.7
    
    # Compute Gaussian basis matrix: shape (n_points, 10)
    # B[i, j] = exp(-(x_i - c_j)^2 / (2 * w^2))
    diff = x_flat[:, np.newaxis] - centers[np.newaxis, :]
    B = np.exp(-diff**2 / (2.0 * w * w))
    
    # Linear combination: y = B @ p
    result = B @ p
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.zeros_like(x_flat)
    
    return result.reshape(orig_shape)
