"""
Rewritten program: High-capacity RBF basis for symbolic regression.

Strategy:
Use a dense set of Gaussian Radial Basis Functions (RBFs) centered at a grid
of points in the [-1, 1] x [-1, 1] domain. The equation is a linear combination
of these RBFs. The 10 parameters are the weights for the 10 selected RBFs.

Why this is different:
- The previous approach used a fixed, low-dimensional polynomial basis (degree 3).
  This limits the model to polynomials, which cannot fit functions like sin(x),
  exp(x), or other non-polynomial shapes.
- The RBF approach uses a high-capacity basis. By placing centers densely
  (e.g., on a 5x5 grid) and using a fixed width, the linear combination of
  10 RBFs can approximate a wide variety of smooth functions.
- The centers and widths are fixed (not learned), which keeps the problem
  linear in the parameters. This ensures that BFGS will converge to the
  global minimum of the linear least-squares problem, avoiding local minima
  and non-convergence issues associated with non-linear parameter optimization.
- The conditioning is managed by normalizing the RBF values or choosing
  appropriate widths. We use a width that provides local support to avoid
  ill-conditioning from overlapping wide Gaussians.

Implementation details:
- Centers: A 5x5 grid on [-1, 1] x [-1, 1]. That's 25 potential centers.
- We need to select 10 of them or use a subset. To ensure coverage, we can
  use a stratified subset or simply the first 10 in a raster scan.
  However, a better approach is to use a set of centers that are well-distributed.
  Let's use a 5x5 grid and pick 10 specific points that cover the domain well.
  Actually, to keep it simple and robust, let's use a 10x10 grid and pick
  every 10th point, or just use a fixed set of 10 well-spread points.
  
  Let's use a 5x5 grid:
  x_centers = [-1, -0.5, 0, 0.5, 1]
  y_centers = [-1, -0.5, 0, 0.5, 1]
  Total 25 points.
  We need 10 parameters. We can use the 10 points that form a "plus" shape
  or a diagonal, or simply the first 10.
  To maximize coverage with only 10 RBFs, we should space them out.
  Let's use a 4x4 grid? No, 4x4=16.
  Let's use a 5x5 grid and select 10 points that are roughly equidistant.
  
  Alternative: Use a single grid of 10 points?
  Let's define 10 centers explicitly:
  (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1), (0, 0), (-0.5, 0.5)
  This covers corners, edges, and center.
  
  Width: sigma = 0.3. This provides local support.
  RBF_i(x, y) = exp(-((x - cx_i)^2 + (y - cy_i)^2) / (2 * sigma^2))
  
  The parameters p[0..9] weight these RBFs.
  f(x, y) = sum(p[i] * RBF_i(x, y))
  
  This is a linear model in p. BFGS will solve it efficiently.
"""
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a fixed RBF basis.
    
    Args:
        x: Input x coordinates.
        y: Input y coordinates.
        params: Array of 10 weights.
        
    Returns:
        Predictions.
    """
    # Define 10 fixed centers in [-1, 1] x [-1, 1]
    # We choose a set that covers the domain reasonably well
    centers = np.array([
        [-1.0, -1.0],
        [-1.0,  0.0],
        [-1.0,  1.0],
        [ 0.0, -1.0],
        [ 0.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  0.0],
        [ 1.0,  1.0],
        [ 0.0,  0.0],
        [-0.5,  0.5]
    ])
    
    # Fixed width for the RBFs
    sigma = 0.3
    sigma_sq = sigma * sigma
    
    # Compute the squared distances from each point (x, y) to each center
    # x and y are arrays of shape (N,)
    # centers is (10, 2)
    
    # Vectorized computation:
    # dx = x[:, None] - centers[None, :, 0]  # shape (N, 10)
    # dy = y[:, None] - centers[None, :, 1]  # shape (N, 10)
    # dist_sq = dx**2 + dy**2                 # shape (N, 10)
    
    dx = x[:, np.newaxis] - centers[np.newaxis, :, 0]
    dy = y[:, np.newaxis] - centers[np.newaxis, :, 1]
    dist_sq = dx**2 + dy**2
    
    # Compute RBF values: exp(-dist_sq / (2 * sigma^2))
    # shape (N, 10)
    rbf_values = np.exp(-dist_sq / (2.0 * sigma_sq))
    
    # Linear combination: f = rbf_values @ params
    # rbf_values is (N, 10), params is (10,)
    # Result is (N,)
    predictions = rbf_values @ params
    
    return predictions
