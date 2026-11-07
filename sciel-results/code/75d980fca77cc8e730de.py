"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a sum of 5 Gaussian Radial Basis Functions.
    params: [c1, s1, c2, s2, c3, s3, c4, s4, c5, s5]
    c_i: amplitude
    s_i: width parameter (used as sigma = s_i^2 + 0.1 to ensure positivity)
    Centers are fixed at -2, -1, 0, 1, 2.
    """
    # Fixed centers
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Extract amplitudes and width parameters
    # params shape is (10,)
    # We reshape to (5, 2) where col 0 is amplitude, col 1 is width param
    p = params.reshape(5, 2)
    amplitudes = p[:, 0]
    width_params = p[:, 1]
    
    # Ensure widths are positive and reasonable
    # Using square + small epsilon to ensure sigma > 0
    sigmas = width_params**2 + 0.1
    
    # Calculate the Gaussian basis functions
    # x shape: (N,)
    # centers shape: (5,)
    # We want result shape (N, 5)
    # x[:, None] - centers[None, :] gives (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian: exp(-diff^2 / (2 * sigma^2))
    # sigma shape: (5,) -> (1, 5) for broadcasting
    sigmas_reshaped = sigmas[np.newaxis, :]
    
    basis = np.exp(-diff**2 / (2.0 * sigmas_reshaped**2))
    
    # Linear combination
    # basis shape: (N, 5)
    # amplitudes shape: (5,)
    result = basis @ amplitudes
    
    return result
# EVOLVE-BLOCK-END
