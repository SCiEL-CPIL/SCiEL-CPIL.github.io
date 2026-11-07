"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF network with 5 fixed centers.
    10 parameters: 5 amplitudes, 5 widths.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Fixed centers evenly spaced across [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Parameters: first 5 are amplitudes, next 5 are widths
    amplitudes = params[:5]
    widths = params[5:]
    
    # Ensure widths are positive to avoid numerical issues
    # Use softplus-like transformation or absolute value with small epsilon
    # BFGS starts at 1, so abs(1)=1 which is fine.
    # Using np.abs ensures width > 0, but gradient is 0 at 0.
    # Better: use np.tanh(width) + 1 to ensure positive? Or just abs.
    # Let's use abs + small constant to avoid division by zero if width becomes 0.
    # Actually, if width is 0, exp(-inf) = 0. That's fine.
    # But if width is negative, exp(-inf) is still 0 if we use abs.
    # Let's just use abs(widths) + 1e-6 to be safe.
    safe_widths = np.abs(widths) + 1e-6
    
    # Compute RBF outputs
    # x has shape (N,), centers has shape (5,)
    # diff has shape (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian: exp(- (x - c)^2 / (2 * width^2))
    # To avoid underflow/overflow, we can clip the argument.
    # But exp of large negative is just 0.
    # exp of large positive is inf. But (x-c)^2 is positive, width^2 is positive.
    # So argument is always negative or zero.
    # So exp will be in (0, 1].
    
    rbf_values = np.exp(-0.5 * (diff / safe_widths[np.newaxis, :])**2)
    
    # Weighted sum
    return rbf_values.dot(amplitudes)

# EVOLVE-BLOCK-END
