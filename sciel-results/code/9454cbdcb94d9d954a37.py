"""
Strategy: Nonlinear Parametric Gaussian Mixture Model.

Uses 5 Gaussian bumps where BOTH the amplitudes and centers are learned
by BFGS. The 10 parameters are split as:
  - params[0:5]  -> amplitudes (a_i)
  - params[5:10] -> centers (mu_i), mapped from raw values via tanh to [-2, 2]

The widths (sigmas) are fixed at 0.35 to keep the problem well-conditioned
while still allowing BFGS to find optimal placement.

Key structural difference from the previous linear RBF:
  - The model is NONLINEAR in parameters (centers appear inside exp()).
  - BFGS must navigate a non-convex landscape, which can find structures
    that a fixed linear basis cannot represent at all.
  - The tanh mapping of centers ensures they stay within the data domain,
    improving conditioning of the optimization.
"""
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Compute the predicted y values using a nonlinear Gaussian mixture model.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 values: [a0..a4, mu0..mu4] where
                a_i are amplitudes and mu_i are tanh-mapped centers.
        
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
    
    # Extract amplitudes and centers
    amps = params[0:5]       # 5 amplitudes
    mu_raw = params[5:10]    # 5 raw center values
    
    # Map centers to [-2, 2] via tanh for stability
    centers = 2.0 * np.tanh(mu_raw)
    
    # Fixed widths
    sigma = 0.35
    sigma2 = 2.0 * sigma ** 2
    
    # Compute Gaussian mixture: y(x) = sum_i a_i * exp(-(x - mu_i)^2 / sigma2)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (N, 5)
    gaussians = np.exp(-(diff ** 2) / sigma2)          # (N, 5)
    result = gaussians @ amps                          # (N,)
    
    # Safety check for numerical stability
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
