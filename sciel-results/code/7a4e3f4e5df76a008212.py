"""Program using a Kernel Ridge Regression strategy with Gaussian RBF kernels."""
import numpy as np
import time

_start_time = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form symbolic regression candidate.
    Uses a linear combination of 10 Gaussian RBF kernels with fixed centers.
    The 10 params act as adaptive weights for local interpolation.
    
    This shifts from global functional basis (polynomials/trig) to local flexibility,
    enabling the model to capture arbitrary smooth functions by weighting local bumps.
    """
    # Check for timeout (though unlikely to trigger for this simple forward pass)
    # If we were doing internal optimization, we would check here.
    # For now, just proceed.
    
    # Scale input to [-1, 1] is not strictly necessary if centers are in [-3, 3],
    # but RBFs work well with normalized inputs.
    # Let's keep x in original scale to match centers in original scale.
    
    # Define centers evenly spaced in [-3, 3]
    # 10 centers
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Define the width (sigma) of the RBFs
    # A sigma of 0.5 provides local flexibility
    sigma = 0.5
    
    # Calculate RBF values
    # x is shape (N,), centers is shape (10,)
    # We want a matrix of shape (N, 10)
    # diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Handle x being a scalar or 1D array
    x_arr = np.asarray(x)
    if x_arr.ndim == 0:
        x_arr = x_arr.reshape(1)
    
    diff = x_arr[:, np.newaxis] - centers[np.newaxis, :]
    
    # RBF: exp(-0.5 * (x - c)^2 / sigma^2)
    # Or exp(-(x-c)^2 / (2*sigma^2))
    rbf_values = np.exp(-0.5 * (diff ** 2) / (sigma ** 2))
    
    # Linear combination with params
    # rbf_values is (N, 10), params is (10,)
    y = rbf_values @ params
    
    return y
