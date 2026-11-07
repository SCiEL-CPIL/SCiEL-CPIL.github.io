"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate an unknown function using a Gaussian RBF basis.
    
    This approach uses 10 Gaussian functions centered evenly across the domain [0.1, 6].
    The model is linear in the parameters, which ensures that the optimization landscape
    is convex (a paraboloid). This allows BFGS to converge rapidly and reliably to the
    global minimum, avoiding the pitfalls of non-linear optimization or ill-conditioned
    polynomial bases.
    """
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Parameters
    x_min = 0.1
    x_max = 6.0
    n_centers = 10
    
    # Define centers evenly spaced
    centers = np.linspace(x_min, x_max, n_centers)
    
    # Define width (sigma). 
    # Spacing between centers is (x_max - x_min) / (n_centers - 1)
    spacing = (x_max - x_min) / (n_centers - 1)
    # Choose sigma to be about 60% of the spacing for good overlap without being too flat
    sigma = 0.6 * spacing
    
    # Compute the RBF basis values
    # x shape: (N,), centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel
    # phi_ij = exp(-0.5 * ((x_i - c_j) / sigma)^2)
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination: y = basis @ params
    # basis shape: (N, 10), params shape: (10,)
    return basis @ params
# EVOLVE-BLOCK-END
