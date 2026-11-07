"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator using a Gaussian RBF basis.
    
    Strategy: 
    1. Normalize inputs x and v to standardize the scale.
    2. Use a fixed set of 10 Gaussian Radial Basis Functions (RBFs) centered 
       at specific points in the (x, v) space.
    3. The parameters 'params' are the linear weights for these RBFs.
    4. This structure makes the optimization problem linear in the parameters,
       which BFGS can solve very efficiently and robustly, avoiding the 
       local minima issues of non-linear polynomial fitting.
    5. The fixed centers are chosen to cover the training range of x and v.
    """
    p = params
    
    # Normalization constants based on training ranges
    # x: [-0.5728, 0.5275] -> mean ~ -0.02, std ~ 0.3
    # v: [-0.1754, 0.2821] -> mean ~ 0.05, std ~ 0.15
    # We use simple scaling to map to roughly [-1, 1] or similar
    x_scale = 0.57
    v_scale = 0.28
    
    x_norm = x / x_scale
    v_norm = v / v_scale
    
    # Define 10 fixed centers in the normalized (x, v) space
    # We want to cover the likely range of inputs.
    # x_norm range approx: [-1.0, 0.9]
    # v_norm range approx: [-0.6, 1.0]
    
    # Grid of centers
    # Let's use a 3x3 grid plus one center, or just 10 well-spaced points.
    # To ensure coverage, we can use a grid.
    # 3 x 3 = 9 points + 1 extra = 10 points.
    
    x_centers = np.array([-0.8, 0.0, 0.8])
    v_centers = np.array([-0.5, 0.0, 0.5])
    
    # Create meshgrid for centers
    cx, cv = np.meshgrid(x_centers, v_centers)
    cx = cx.flatten()
    cv = cv.flatten()
    
    # We have 9 points. Add a 10th point at the origin to capture the mean/bias effectively.
    cx = np.append(cx, 0.0)
    cv = np.append(cv, 0.0)
    
    # Width of the Gaussians. 
    # If width is too small, it's too local (underfitting with only 10 points).
    # If width is too large, it's too global (similar to polynomial).
    # A moderate width allows local adaptation.
    # Distance in normalized space. Max distance is ~ sqrt(1^2 + 1^2) = 1.4.
    # sigma = 0.5 seems reasonable to allow overlap but local specificity.
    sigma = 0.5
    
    # Compute RBF values
    # dx = x_norm[:, None] - cx[None, :]
    # dv = v_norm[:, None] - cv[None, :]
    # dist_sq = dx**2 + dv**2
    # rbf_vals = np.exp(-dist_sq / (2 * sigma**2))
    
    # Efficient computation
    dx = x_norm[:, None] - cx[None, :]
    dv = v_norm[:, None] - cv[None, :]
    dist_sq = dx**2 + dv**2
    rbf_vals = np.exp(-dist_sq / (2 * sigma**2))
    
    # rbf_vals shape: (N, 10)
    # params shape: (10,)
    
    # Prediction is the dot product
    # Ensure params is flattened
    p_flat = p.flatten()
    
    return np.sum(rbf_vals * p_flat, axis=1)

# EVOLVE-BLOCK-END
