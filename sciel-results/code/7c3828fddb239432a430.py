import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a fixed-center Gaussian Radial Basis Function (RBF) basis.
    This approach uses 10 fixed Gaussian centers distributed uniformly over the input domain [-2, 2].
    The parameters are the weights for each RBF. This linear-in-parameters model allows
    BFGS to efficiently find the optimal weights, providing a flexible non-polynomial approximation.
    """
    # Fixed centers for the 10 RBFs
    centers = np.linspace(-2.0, 2.0, 10)
    # Fixed width (sigma) for the Gaussians
    sigma = 1.0
    
    # Compute the RBF basis matrix
    # shape: (10, N)
    # R[i, j] = exp(-((x_j - c_i) / sigma)^2)
    x_col = x[:, np.newaxis]
    c_row = centers[np.newaxis, :]
    
    diff = (x_col - c_row) / sigma
    R = np.exp(-diff**2)
    
    # Linear combination
    return np.dot(params, R)
