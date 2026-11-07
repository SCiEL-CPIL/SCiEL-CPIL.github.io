"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input to [-1, 1] for better conditioning
    # x is in [-3, 3], so u = x / 3 is in [-1, 1]
    u = x / 3.0
    
    # Use Gaussian RBF kernels with fixed centers
    # This allows local flexibility for the 10 parameters
    # Centers evenly spaced in [-1, 1]
    centers = np.linspace(-1.0, 1.0, 10)
    
    # Bandwidth for RBF kernels - tuned for good local coverage
    # Smaller bandwidth = more local, larger = smoother
    bandwidth = 0.4
    
    # Compute RBF basis values
    # Each term is exp(-((u - center)^2) / (2 * bandwidth^2))
    # This creates a set of local "bumps" that can be weighted by params
    
    # Vectorized computation: u is (N,), centers is (10,)
    # diff is (N, 10)
    diff = u[:, np.newaxis] - centers[np.newaxis, :]
    
    # RBF values: (N, 10)
    rbf_values = np.exp(-diff**2 / (2.0 * bandwidth**2))
    
    # Combine with params: rbf_values @ params gives (N,)
    return rbf_values @ params

# EVOLVE-BLOCK-END
