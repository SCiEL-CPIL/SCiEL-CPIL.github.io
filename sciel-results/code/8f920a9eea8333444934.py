import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sum of 5 Gaussian bumps with fixed centers at 1, 2, 3, 4, 5.
    Params 0-4: Amplitudes A_0 ... A_4
    Params 5-9: Widths (sigma) for each Gaussian, clipped to [0.2, 5.0]
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Fixed centers
    centers = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Extract amplitudes and widths
    amplitudes = params[0:5]
    widths = params[5:10]
    
    # Clip widths to ensure stability and avoid singularities
    widths = np.clip(widths, 0.2, 5.0)
    
    # Compute Gaussians
    # (x - centers)^2 / (2 * widths^2)
    # Shape of x: (N,)
    # Shape of centers: (5,)
    # We need to broadcast.
    # diff = x[:, None] - centers[None, :] -> (N, 5)
    
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    exponents = - (diff ** 2) / (2.0 * widths[np.newaxis, :] ** 2)
    gaussians = np.exp(exponents)
    
    # Weighted sum
    # y = sum(amplitudes[i] * gaussians[:, i])
    y = np.sum(gaussians * amplitudes[np.newaxis, :], axis=1)
    
    return y
