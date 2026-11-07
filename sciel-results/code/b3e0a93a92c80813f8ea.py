import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Approach: Gaussian Radial Basis Functions (RBF)
    # This is a different family from Chebyshev polynomials.
    # RBFs are local approximators and form a positive-definite kernel.
    # The model is linear in the parameters (weights), ensuring convex optimization
    # for BFGS.
    
    # Define 10 fixed centers evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Define a fixed width (sigma) for the Gaussians.
    # A width of 0.5 to 1.0 is typical for this range.
    # We choose 0.6 to ensure good overlap between neighbors for smoothness.
    sigma = 0.6
    
    # Compute the RBF features
    # x shape: (N,)
    # centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Calculate Gaussian values: exp(-0.5 * (x-c)^2 / sigma^2)
    # Note: Using 2*sigma^2 in denominator for standard normal form, or just sigma^2.
    # Let's use exp(-(x-c)^2 / (2*sigma^2))
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination: y = rbf_values @ params
    # rbf_values shape: (N, 10)
    # params shape: (10,)
    y = rbf_values @ params
    
    return y
