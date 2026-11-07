"""Symbolic regression with 2 inputs: Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Radial Basis Function (RBF) model with 10 parameters.
    
    The model is:
    f(x, y) = p_0 + sum_{i=1}^{9} p_i * exp(-((x - c_{x,i})^2 + (y - c_{y,i})^2) / (2 * sigma^2))
    
    Where:
    - p_0 is a constant bias term.
    - The 9 RBF centers are on a 3x3 grid in [-1, 1]x[-1, 1].
    - sigma is the width of the Gaussian kernels.
    
    This basis provides a universal approximator for smooth functions on the domain,
    avoiding the limitations of low-degree polynomial bases and the instability
    of rational bases. The model is linear in parameters, allowing BFGS to
    converge reliably to the global minimum.
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Define the 9 RBF centers on a 3x3 grid
    # Using 0.9 to cover most of the [-1, 1] range effectively
    centers_x = np.array([-0.9, 0.0, 0.9])
    centers_y = np.array([-0.9, 0.0, 0.9])
    
    # Define the Gaussian width (sigma)
    # sigma = 0.6 provides a balance between local specificity and global smoothness
    sigma = 0.6
    sigma_sq = 2.0 * sigma * sigma
    
    # Initialize the basis matrix with the constant term
    # Shape: (N, 10)
    n_points = len(x)
    basis = np.ones((n_points, 10), dtype=np.float64)
    
    # Fill the basis with Gaussian RBFs
    idx = 1
    for cx in centers_x:
        for cy in centers_y:
            # Compute squared distances to the center (cx, cy)
            # dist_sq = (x - cx)^2 + (y - cy)^2
            dx = x - cx
            dy = y - cy
            dist_sq = dx * dx + dy * dy
            
            # Compute Gaussian values
            # Using exp(-dist_sq / sigma_sq)
            # We can use np.exp for clarity
            basis[:, idx] = np.exp(-dist_sq / sigma_sq)
            idx += 1
            
    # Compute the prediction: linear combination of basis functions
    # params is of length 10
    pred = basis @ params
    
    # Ensure finite values (though Gaussians are always finite, this is a safety check)
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
