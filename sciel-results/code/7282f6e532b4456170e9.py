"""Symbolic regression: recover hidden function via RBF basis expansion fit by BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (Gaussian) model with 5 fixed centers.
    params: array of length 10
    - params[0:5]: weights for the 5 Gaussian RBFs
    - params[5:10]: log-widths for the 5 Gaussian RBFs (exponentiated to get widths)
    
    Centers are fixed at evenly spaced points across [-3, 3]:
    c_i = -3 + 6*i/4 for i=0..4 => [-3, -1.5, 0, 1.5, 3]
    
    The model is linear in the weights (first 5 params), but nonlinear in widths.
    BFGS can handle this since the objective is smooth.
    """
    # Extract weights and log-widths
    weights = params[0:5]
    log_widths = params[5:10]
    
    # Convert log-widths to actual widths (exp to ensure positivity)
    # Clip to avoid overflow/underflow
    widths = np.exp(np.clip(log_widths, -5.0, 5.0))
    
    # Fixed centers evenly spaced in [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Compute RBF basis: phi_i(x) = exp(-((x - c_i)^2) / (2 * w_i^2))
    result = np.zeros_like(x)
    for i in range(5):
        diff = x - centers[i]
        result += weights[i] * np.exp(-(diff * diff) / (2.0 * widths[i] * widths[i]))
    
    return result
# EVOLVE-BLOCK-END
