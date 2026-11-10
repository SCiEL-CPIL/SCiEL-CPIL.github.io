"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of fixed-center 
    Gaussian basis functions (RBF).
    
    The model is: y(x) = sum_{i=0}^{9} p_i * exp(-(x - c_i)^2 / (2 * sigma^2))
    
    This model is linear in the parameters p_i, which allows scipy.optimize.minimize 
    with BFGS to converge rapidly to the optimal solution for the least-squares problem.
    """
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64)
    
    # Fixed centers spread uniformly across the input domain [-2, 2]
    # 10 centers corresponding to the 10 parameters
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed width (standard deviation) of the Gaussian kernels
    # Chosen to be roughly the spacing between centers to allow for smooth interpolation
    sigma = 0.5
    
    # Compute the Gaussian basis matrix
    # x has shape (N,), centers has shape (10,)
    # We want result shape (N, 10)
    # x[:, np.newaxis] - centers[np.newaxis, :] gives (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute the exponent part
    # exp(-diff^2 / (2 * sigma^2))
    basis = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    
    # Linear combination of basis functions with the fitted parameters
    # basis has shape (N, 10), params has shape (10,)
    # Result has shape (N,)
    prediction = basis @ params
    
    return prediction
# EVOLVE-BLOCK-END
