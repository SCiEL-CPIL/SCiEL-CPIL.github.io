"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using Gaussian Radial Basis Functions (RBFs).
    
    The model uses a linear combination of 9 Gaussian RBFs centered on a grid
    in the normalized (x, v) space, plus a bias term. This makes the model
    linear in parameters, ensuring that BFGS converges to the global minimum
    of the MSE, avoiding local minima issues associated with polynomial fits.
    
    Inputs are normalized to [-1, 1] to improve numerical stability.
    """
    # Normalization constants based on training ranges
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    x_range = x_max - x_min
    v_range = v_max - v_min
    
    # Scale to [-1, 1]
    x_norm = 2.0 * (x - x_min) / x_range - 1.0
    v_norm = 2.0 * (v - v_min) / v_range - 1.0
    
    # Define 9 centers in a 3x3 grid in normalized space
    # Centers are slightly inside the [-1, 1] bounds
    centers_x = np.array([-0.6, 0.0, 0.6])
    centers_v = np.array([-0.6, 0.0, 0.6])
    
    # Create meshgrid of centers
    cx_grid, cv_grid = np.meshgrid(centers_x, centers_v, indexing='ij')
    cx_flat = cx_grid.flatten()
    cv_flat = cv_grid.flatten()
    
    # Compute squared distances from each point to each center
    # x_norm shape: (N,), cx_flat shape: (9,)
    # diff_x shape: (N, 9)
    diff_x = x_norm[:, np.newaxis] - cx_flat[np.newaxis, :]
    diff_v = v_norm[:, np.newaxis] - cv_flat[np.newaxis, :]
    
    # Squared distance
    dist_sq = diff_x**2 + diff_v**2
    
    # Gaussian RBF with fixed width sigma
    # Sigma chosen to be wide enough to capture global structure but narrow enough for local detail
    sigma = 0.5
    rbf_values = np.exp(-dist_sq / (2.0 * sigma**2))
    
    # rbf_values shape: (N, 9)
    # params[1:10] are the weights for the 9 RBFs
    # params[0] is the bias
    
    bias = params[0]
    weights = params[1:10]
    
    # Linear combination
    pred = rbf_values @ weights + bias
    
    return pred
# EVOLVE-BLOCK-END
