"""Kernel Ridge Regression with adaptive Gaussian RBF centers via parameterization."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Kernel Ridge Regression with 5 Gaussian RBF kernels.
    
    The 10 parameters are split into:
    - params[0:5]  -> kernel centers (mapped to [-3, 3] via tanh)
    - params[5:10] -> kernel weights
    
    The bandwidth (gamma) is fixed to allow BFGS to adapt centers and weights.
    This gives the model more local flexibility than fixed evenly-spaced centers.
    """
    # Split parameters: 5 centers, 5 weights
    center_params = params[:5]
    weights = params[5:]
    
    # Map center parameters from R to [-3, 3] using tanh scaling
    # tanh outputs (-1, 1), so scale by 3 to get (-3, 3)
    centers = 3.0 * np.tanh(center_params)
    
    # Fixed bandwidth for the Gaussian kernels
    # Using sigma = 1.0 gives gamma = 1/(2*sigma^2) = 0.5
    # This is a reasonable width for local interpolation over [-3, 3]
    gamma = 0.5
    
    # Compute distances from each input point to each center
    # x shape: (n,), centers shape: (5,)
    # diff shape: (n, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute Gaussian RBF values
    # rbf shape: (n, 5)
    rbf = np.exp(-gamma * diff ** 2)
    
    # Linear combination with weights
    # prediction shape: (n,)
    prediction = rbf @ weights
    
    # Ensure finite output
    prediction = np.where(np.isfinite(prediction), prediction, 0.0)
    
    return prediction
# EVOLVE-BLOCK-END
