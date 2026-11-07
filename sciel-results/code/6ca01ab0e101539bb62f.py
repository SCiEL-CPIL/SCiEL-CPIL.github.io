"""RBF network with 5 fixed centers for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) network with 5 fixed, evenly spaced centers.
    
    params[0:5]  -> amplitudes for the 5 Gaussian kernels
    params[5:10] -> widths (inverse scale) for the 5 Gaussian kernels
    
    f(x) = sum_{i=0}^{4} a_i * exp(-w_i * (x - c_i)^2)
    
    Centers c_i are fixed at evenly spaced points in [-3, 3]:
    c_i = -3 + 6 * i / 4 for i = 0, 1, 2, 3, 4
    i.e., c = [-3, -1.5, 0, 1.5, 3]
    
    Widths are parameterized as w_i = params[5+i], and we use
    exp(-w_i * (x - c_i)^2) so that larger w_i means narrower kernel.
    To ensure numerical stability and avoid degenerate solutions,
    we clip widths to a reasonable range.
    """
    # Fixed centers evenly spaced across [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0], dtype=np.float64)
    
    # Extract amplitudes and widths
    amplitudes = params[0:5]
    widths = params[5:10]
    
    # Clip widths to avoid numerical issues (too small -> nearly constant, too large -> spike)
    # We want widths to be positive; clip to [0.01, 100]
    widths = np.clip(widths, 0.01, 100.0)
    
    # Compute RBF sum: f(x) = sum_i a_i * exp(-w_i * (x - c_i)^2)
    # x shape: (N,), centers shape: (5,)
    # We need (N, 5) for (x - c_i)^2
    x_col = x[:, np.newaxis]  # (N, 1)
    c_row = centers[np.newaxis, :]  # (1, 5)
    
    # Squared distances
    sq_dist = (x_col - c_row) ** 2  # (N, 5)
    
    # Apply widths: -w_i * sq_dist
    # widths shape: (5,) -> (1, 5)
    w_row = widths[np.newaxis, :]  # (1, 5)
    
    # Exponentials
    rbf_vals = np.exp(-w_row * sq_dist)  # (N, 5)
    
    # Weighted sum
    # amplitudes shape: (5,) -> (1, 5)
    a_row = amplitudes[np.newaxis, :]  # (1, 5)
    
    result = rbf_vals @ amplitudes  # (N,)
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
