"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Global Radial Basis Function (RBF) Basis Expansion.
    
    Strategy:
    Instead of a non-linear neural network (MLP) which is hard to optimize with BFGS 
    (non-convex loss), we use a set of fixed RBF centers. The model is linear in 
    the parameters (weights), so BFGS will converge to the global minimum of the 
    linear least squares problem, effectively performing a linear regression in a 
    high-dimensional feature space defined by the RBFs.
    
    Architecture:
    - Input: 2 features (x, y)
    - Basis: 9 fixed RBF centers distributed across the domain [-1, 1]x[-1, 1]
    - Output: Linear combination of RBF activations + global bias
    
    Parameters (10 total):
    p0..p8: Weights for the 9 RBF centers
    p9: Global bias term
    
    Centers:
    A 3x3 grid of points: {-0.6, 0, 0.6} for both x and y.
    This provides coverage of the corners, edges, and center of the domain.
    
    RBF Definition:
    phi_i(x, y) = exp(-beta * ||[x, y] - c_i||^2)
    We fix beta = 2.0 to balance locality and global coverage. 
    (Too small: behaves like a constant/bias. Too large: becomes very local/underfitting 
    with only 9 params. 2.0 allows overlap for smooth interpolation).
    
    Model:
    result = sum(p_i * phi_i(x, y)) + p9
    """
    if params is None or len(params) < 10:
        return np.zeros_like(x)
    
    # Extract parameters
    weights = params[:9]
    bias = params[9]
    
    # Define fixed centers (3x3 grid)
    # Using -0.6, 0, 0.6 to avoid exact boundary issues and provide good spread
    centers = [
        (-0.6, -0.6), (0.0, -0.6), (0.6, -0.6),
        (-0.6, 0.0),  (0.0, 0.0),  (0.6, 0.0),
        (-0.6, 0.6),  (0.0, 0.6),  (0.6, 0.6)
    ]
    
    # RBF width parameter (fixed)
    beta = 2.0
    
    result = np.full_like(x, bias, dtype=float)
    
    # Compute contribution of each RBF
    for i, (cx, cy) in enumerate(centers):
        # Calculate squared distance
        dx = x - cx
        dy = y - cy
        dist_sq = dx * dx + dy * dy
        
        # Compute RBF activation
        # exp(-beta * dist_sq)
        # Clip arguments to prevent overflow/underflow issues, though exp handles it well
        # dist_sq is non-negative. Max dist_sq is approx (1.6)^2 + (1.6)^2 = 5.12
        # beta * dist_sq max is ~10.24. exp(-10) is small but non-zero.
        rbf_val = np.exp(-beta * dist_sq)
        
        result += weights[i] * rbf_val
        
    return result
# EVOLVE-BLOCK-END
