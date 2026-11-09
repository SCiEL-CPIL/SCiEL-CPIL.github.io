"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using Gaussian Radial Basis Functions (RBFs).
    params: array of length 10.
    
    We use 10 Gaussian RBFs with fixed centers and widths, and learnable amplitudes.
    Centers are uniformly distributed across the domain [-3, 3].
    Widths are fixed to provide local flexibility.
    
    This approach is distinct from polynomial bases and handles smooth non-periodic
    functions well. The model is linear in parameters, so BFGS should converge
    robustly to the least-squares solution.
    """
    # Ensure x is 1D array
    x = np.asarray(x).flatten()
    # Ensure params is 1D array
    p = np.asarray(params).flatten()
    
    # Define fixed RBF centers
    # 10 centers uniformly spaced in [-3, 3]
    # Using midpoints of 10 equal intervals
    centers = np.linspace(-3.0, 3.0, 11)
    c = (centers[:-1] + centers[1:]) / 2.0
    
    # Fixed width parameter
    # sigma_sq controls the spread. 
    # Spacing between centers is 0.6.
    # A sigma_sq of 0.4 (sigma ~ 0.63) provides good overlap.
    sigma_sq = 0.4 
    
    # Compute RBF values
    # x shape: (N,)
    # c shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - c[np.newaxis, :]
    
    # exp(-diff^2 / (2 * sigma_sq))
    rbf_values = np.exp(-diff**2 / (2.0 * sigma_sq))
    
    # Prediction: rbf_values @ p
    # rbf_values shape: (N, 10)
    # p shape: (10,)
    pred = rbf_values @ p
    
    return pred
# EVOLVE-BLOCK-END
