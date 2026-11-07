"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a fixed-center Radial Basis Function (RBF) network with 10 Gaussian centers.
    The centers are evenly spaced across the input domain [0.1, 6.0].
    The parameters p0..p9 control the amplitude of each Gaussian bump.
    This allows the model to approximate local features of the hidden function
    without relying on a specific global functional form.
    """
    # Ensure x is a numpy array for safe operations
    x = np.asarray(x, dtype=np.float64)
    
    # Define 10 fixed centers evenly spaced in [0.1, 6.0]
    # We use np.linspace to generate them
    centers = np.linspace(0.1, 6.0, 10)
    
    # Extract parameters
    # p0..p9 correspond to the weights of the 10 RBFs
    
    # Compute the Gaussian basis functions
    # We choose a fixed width (sigma) for all Gaussians.
    # A smaller sigma allows for sharper local features, but might require more parameters.
    # With 10 parameters and a range of ~6, a sigma around 0.8-1.0 provides good coverage
    # with some overlap, allowing smooth interpolation.
    sigma = 0.8
    
    # Vectorized computation for efficiency
    # x is shape (N,), centers is shape (10,)
    # diff is shape (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute exp(-0.5 * (diff/sigma)^2)
    # This is the standard Gaussian RBF
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination: y = basis @ params
    # basis is (N, 10), params is (10,)
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
