"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# Fixed hyperparameters for the RBF basis
# Domain is [0.1, 6]
# We use 10 RBF centers evenly spaced to maximize local flexibility
# using all 10 parameters as weights.
_CENTERS = np.linspace(0.1, 6.0, 10)

# Define a bandwidth that allows for good overlap between adjacent centers
# Distance between centers is approx (6.0 - 0.1) / 9 = 0.655
# A sigma of 0.6 ensures significant overlap, aiding smooth interpolation
_SIGMA = 0.6
_INV_2_SIGMA2 = 1.0 / (2.0 * _SIGMA * _SIGMA)


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the model prediction using a 10-center Gaussian RBF basis.
    
    Model: y = sum(p[i] * RBF(x, c_i)) for i in 0..9
    
    Args:
    x: Input array of shape (N,)
    params: Parameter array of shape (10,)
        - params[0:10]: Weights for the 10 RBFs
    
    Returns:
    Prediction array of shape (N,)
    """
    # Ensure x is a 1D array for calculation
    x_flat = np.asarray(x).flatten()
    
    # Compute differences between x and each center
    # x_flat shape: (N,)
    # _CENTERS shape: (10,)
    # diff shape: (N, 10)
    diff = x_flat[:, np.newaxis] - _CENTERS[np.newaxis, :]
    
    # Compute RBF values
    # exp_arg shape: (N, 10)
    exp_arg = -diff * diff * _INV_2_SIGMA2
    rbf_values = np.exp(exp_arg) # Shape (N, 10)
    
    # Linear combination of RBFs
    # rbf_values @ params gives shape (N,)
    y = rbf_values @ params
    
    return y
# EVOLVE-BLOCK-END
