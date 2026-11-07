"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a linear combination of 10 Gaussian basis functions.
    
    Centers are evenly spaced across [-3, 3].
    Widths are fixed to provide local flexibility without ill-conditioning.
    params[0:10] are the weights for each Gaussian basis function.
    
    This acts as a universal approximator that is well-conditioned for BFGS,
    avoiding the issues of high-degree global polynomials.
    """
    # Ensure x is 1D for consistent shape handling
    x = np.asarray(x, dtype=float)
    original_shape = x.shape
    x_flat = x.ravel()
    
    # 10 basis functions centered evenly across [-3, 3]
    # Centers: -3, -2.333, -1.666, -1, -0.333, 0.333, 1, 1.666, 2.333, 3
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Fixed widths: chosen to cover the interval with overlap but maintain locality.
    # A width of ~0.6 provides better local sensitivity while maintaining conditioning.
    widths = np.full(10, 0.6)
    
    # Compute Gaussian basis values: exp(-((x - center)^2) / (2 * width^2))
    # Shape of x_flat: (N,)
    # Shape of centers: (10,)
    # We want a matrix of shape (N, 10) where each column is one basis function evaluated at all x points.
    
    diff = x_flat[:, np.newaxis] - centers[np.newaxis, :]  # (N, 10)
    basis = np.exp(-0.5 * (diff / widths[np.newaxis, :]) ** 2)  # (N, 10)
    
    # params[0:10] are the weights
    weights = params[0:10]
    
    # Linear combination
    prediction_flat = basis @ weights  # (N,)
    
    # Restore original shape
    prediction = prediction_flat.reshape(original_shape)
    
    return prediction
# EVOLVE-BLOCK-END
