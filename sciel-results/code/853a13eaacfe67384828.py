"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate an unknown function using a linear combination of 10 Radial Basis Functions (RBFs)
    at fixed, evenly spaced centers, with a single learnable width parameter shared by all kernels.
    
    params[0:9] are the weights for each RBF center.
    params[9] is the inverse width (beta) for the Gaussian RBFs.
    
    Centers are fixed at evenly spaced points in [-3, 3].
    """
    # Extract weights and width parameter
    weights = params[:9]
    beta = params[9]
    
    # Ensure beta is positive to avoid numerical issues (Gaussian RBF width)
    # We use a softplus-like transformation or just clamp to a small positive value
    # Since BFGS can go negative, we use abs or a minimum threshold
    # Using a minimum of 0.1 to prevent division by zero or extremely narrow peaks
    beta_safe = np.maximum(np.abs(beta), 0.1)
    
    # Fixed centers evenly spaced in [-3, 3]
    # 9 centers (since we have 9 weights)
    centers = np.linspace(-3.0, 3.0, 9)
    
    # Compute RBF values: exp(-beta^2 * (x - c)^2)
    # x shape: (n,), centers shape: (9,)
    # We need (n, 9) matrix of distances
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (n, 9)
    squared_diff = diff ** 2
    
    # Compute Gaussian RBFs
    rbf_values = np.exp(-(beta_safe ** 2) * squared_diff)  # (n, 9)
    
    # Linear combination
    prediction = rbf_values @ weights
    
    return prediction
# EVOLVE-BLOCK-END
