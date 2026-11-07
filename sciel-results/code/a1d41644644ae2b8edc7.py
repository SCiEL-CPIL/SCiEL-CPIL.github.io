"""New program for symbolic regression using Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Gaussian Radial Basis Function (RBF) basis.
    
    This model approximates the target function as a linear combination of
    10 Gaussian kernels centered at equally spaced points in the domain [0.1, 6].
    
    f(x) = sum_{i=0}^{9} w_i * exp(-(x - c_i)^2 / (2 * sigma^2))
    
    This approach uses local basis functions, which can better capture
    localized features or non-polynomial smooth shapes compared to global
    polynomial bases. The model is linear in parameters, ensuring stable
    optimization with BFGS.
    
    params (length 10) are the weights w_i for the Gaussian kernels.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Define centers for the RBFs in the domain [0.1, 6]
    min_x = 0.1
    max_x = 6.0
    n_centers = 10
    
    # Equally spaced centers
    centers = np.linspace(min_x, max_x, n_centers)
    
    # Choose a bandwidth sigma that covers the domain well.
    # A sigma around 1.0-1.5 is typically effective for this domain width (~5.9).
    # We use 1.2 as a balanced choice.
    sigma = 1.2
    sigma_sq = 2.0 * sigma * sigma
    
    # Compute the RBF values for all x and all centers
    # x is shape (N,), centers is shape (M,)
    # diff is shape (N, M)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute exp(-diff^2 / sigma_sq)
    # Note: We use the standard Gaussian form exp(-0.5 * (x-c)^2 / sigma^2)
    # So the exponent is -diff^2 / (2 * sigma^2)
    rbf_values = np.exp(-diff**2 / sigma_sq)
    
    # Linear combination: result = rbf_values @ params
    # rbf_values is (N, 10), params is (10,)
    result = rbf_values @ params
    
    return result
# EVOLVE-BLOCK-END
