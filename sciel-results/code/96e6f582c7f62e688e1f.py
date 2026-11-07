"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Flatten input to ensure 1D processing
    x_flat = np.asarray(x, dtype=np.float64).flatten()
    n_points = x_flat.shape[0]
    
    if n_points == 0:
        return np.array([])
        
    # Ensure params is 1D
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Strategy: Linear combination of 9 RBFs + 1 learnable width
    # params[0:9] are weights for 9 RBFs
    # params[9] is the width parameter
    
    # Define 9 fixed centers evenly spaced in [-3, 3]
    # Centers: -3, -2.25, -1.5, -0.75, 0, 0.75, 1.5, 2.25, 3
    centers = np.linspace(-3.0, 3.0, 9)
    
    # Extract width parameter
    # Use a robust transformation to ensure positive width
    # Start value is 1.0, so abs(1)+0.1 = 1.1
    # If param becomes negative, abs handles it.
    # Adding 0.1 prevents division by zero.
    sigma = np.abs(p[9]) + 0.1
    
    # Precompute RBF matrix
    # Shape: (n_points, 9)
    # phi(x, c) = exp(-(x-c)^2 / (2 * sigma^2))
    
    # To avoid overflow/underflow issues in exp, we can clip the argument
    # Argument is always <= 0, so exp is in (0, 1]. No overflow.
    # Underflow to 0 is fine.
    
    # Vectorized computation
    # x_flat shape (n_points, 1)
    # centers shape (1, 9)
    diff = x_flat[:, np.newaxis] - centers[np.newaxis, :]
    
    # Calculate exponent
    exponent = -(diff ** 2) / (2.0 * sigma ** 2)
    
    # Compute RBF values
    R = np.exp(exponent)
    
    # Linear combination
    # weights are p[0:9]
    weights = p[0:9]
    
    # Result
    result = R @ weights
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
# EVOLVE-BLOCK-END
