import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params length 10:
    # params[0:3] -> centers (mu_0, mu_1, mu_2)
    # params[3:6] -> widths (sigma_0, sigma_1, sigma_2)
    # params[6:10] -> amplitudes (a_0, a_1, a_2, a_3)
    
    if len(x) == 0:
        return np.array([])
        
    # Extract parameters
    mu = params[0:3]
    sigma = params[3:6]
    a = params[6:10]
    
    # Safety margins to prevent numerical overflow/underflow in exp and division
    # Widths must be positive. If BFGS pushes them to 0 or negative, the Gaussian becomes a Dirac delta or invalid.
    # We clamp sigma to a small positive value.
    sigma = np.maximum(sigma, 1e-3)
    
    # Compute Gaussian basis functions
    # y = sum_{i=0}^{2} a_i * exp(-(x - mu_i)^2 / (2 * sigma_i^2)) + a_3
    # Note: a_3 acts as a global offset/bias term (equivalent to a Gaussian with infinite width, or just a constant).
    
    # Vectorized computation
    # x shape: (N, 1) or (N,)
    # We need to compute for each center i, the value for all x.
    
    # Reshape for broadcasting: x is (N, 1), mu is (1, 3), sigma is (1, 3)
    x_col = x.reshape(-1, 1)
    mu_row = mu.reshape(1, -1)
    sigma_row = sigma.reshape(1, -1)
    
    # Differences
    diff = x_col - mu_row  # (N, 3)
    
    # Exponent term
    # We want exp(-diff^2 / (2 * sigma^2))
    # To avoid overflow in the exponent (if sigma is very small, diff^2/sigma^2 can be huge),
    # we rely on the clamping of sigma. Even with sigma=1e-3, if diff is large, the exponent is very negative,
    # which underflows to 0 safely in float64.
    
    exponent = -0.5 * (diff / sigma_row) ** 2
    basis = np.exp(exponent)  # (N, 3)
    
    # Linear combination of the 3 Gaussians
    # a[0:3] corresponds to the 3 Gaussians
    y_gauss = basis @ a[0:3]  # (N,)
    
    # Add the bias term
    y = y_gauss + a[3]
    
    return y
