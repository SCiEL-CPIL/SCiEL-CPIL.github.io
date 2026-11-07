import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a sum of Gaussian Radial Basis Functions.
    params[0:5] are amplitudes.
    params[5:10] are log-widths (sigma = exp(log_width)).
    Centers are fixed at -1.6, -0.8, 0, 0.8, 1.6.
    """
    # Fixed centers spread across the domain [-2, 2]
    centers = np.array([-1.6, -0.8, 0.0, 0.8, 1.6])
    
    # Extract amplitudes and log-widths
    amplitudes = params[0:5]
    log_widths = params[5:10]
    
    # Calculate widths, ensuring they are positive
    # exp(log_width) is always positive. 
    # If log_width is very negative, sigma -> 0, which can cause numerical issues in division or exp overflow if not careful.
    # However, exp(- (x-c)^2 / (2*sigma^2)) with small sigma is 0 everywhere except at c.
    # To be safe against sigma being too small, we can clip the exponent or use a minimum sigma.
    # Let's just compute it. NumPy handles exp of large negative numbers well (returns 0).
    widths = np.exp(log_widths)
    
    # Avoid division by zero if width is 0 (though exp never returns 0, it can underflow to 0 for very large negative inputs)
    # Add a tiny epsilon to widths to be safe
    widths = np.maximum(widths, 1e-6)
    
    # Compute the basis functions
    # x shape (N,), centers shape (5,)
    # We want to compute exp(-(x[:, None] - centers[None, :])^2 / (2 * widths[None, :]^2))
    
    x_expanded = x[:, np.newaxis] # (N, 1)
    c_expanded = centers[np.newaxis, :] # (1, 5)
    w_expanded = widths[np.newaxis, :] # (1, 5)
    
    # Compute squared distances
    diff = x_expanded - c_expanded # (N, 5)
    sq_dist = diff * diff # (N, 5)
    
    # Compute exponents
    # Denominator: 2 * w^2
    denom = 2.0 * (w_expanded * w_expanded) # (1, 5)
    exponents = -sq_dist / denom # (N, 5)
    
    # Compute Gaussians
    gaussians = np.exp(exponents) # (N, 5)
    
    # Weighted sum
    # amplitudes shape (5,) -> (1, 5)
    result = gaussians @ amplitudes # (N,)
    
    return result
