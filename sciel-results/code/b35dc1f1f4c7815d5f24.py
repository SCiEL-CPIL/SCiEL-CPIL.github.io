# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a linear combination of 10 Radial Basis Functions (RBFs)
    at fixed, evenly spaced centers, with a single learnable width parameter shared by all kernels.
    
    The 10 params are used as:
    - params[0:9] : weights for the 9 RBF kernels at fixed centers
    - params[9]   : shared width parameter (squared) for all RBF kernels
    
    The centers are fixed at evenly spaced points in [-3, 3]:
    centers = [-2.7, -1.8, -0.9, 0.0, 0.9, 1.8, 2.7] plus two more to make 10? 
    Wait, we have 10 params. Let's use 9 RBF weights + 1 width = 10 params.
    So 9 centers.
    Centers: np.linspace(-3, 3, 9) = [-3.0, -2.25, -1.5, -0.75, 0.0, 0.75, 1.5, 2.25, 3.0]
    
    This structure is:
    1. Well-conditioned: The RBF matrix with a single shared width is typically well-conditioned
       for BFGS, especially when the width is learned but shared.
    2. Smooth: RBFs are infinitely differentiable, allowing BFGS to navigate the landscape easily.
    3. Flexible: 9 centers with a learnable width can approximate a wide variety of smooth functions.
    4. Efficient: Only 10 parameters to optimize, and the computation is vectorized.
    """
    t0 = time.time()
    
    x = np.asarray(x, dtype=float).flatten()
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    n_points = len(x)
    if n_points == 0:
        return np.array([])
        
    # Extract parameters
    weights = params[0:9]  # 9 RBF weights
    width_sq = params[9]   # 1 shared width parameter (squared)
    
    # Fixed centers: 9 evenly spaced points in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 9)
    
    # Ensure width is positive and reasonable to avoid numerical issues
    # We use width_sq directly, but clip it to a reasonable range
    # If width_sq is negative, we use its absolute value or a small positive number
    # To keep it differentiable, we can use: effective_width = width_sq**2 + epsilon
    # But the directive says "single learnable width parameter shared by all kernels"
    # Let's interpret width_sq as the actual width parameter, and square it to ensure positivity
    # Or, better, let width_sq be the parameter, and use exp(-width_sq * (x-c)^2)
    # To ensure width_sq > 0, we can use: width = np.exp(width_sq) or width = width_sq**2 + eps
    # Using width_sq**2 + eps ensures differentiability and positivity
    eps = 1e-8
    effective_width = width_sq**2 + eps
    
    # Compute RBF values
    # x has shape (n_points,), centers has shape (9,)
    # We want a matrix of shape (n_points, 9) where entry (i, j) is exp(-effective_width * (x[i] - centers[j])**2)
    
    # Vectorized computation:
    # x_col = x[:, np.newaxis]  # (n_points, 1)
    # centers_row = centers[np.newaxis, :]  # (1, 9)
    # dist_sq = (x_col - centers_row)**2  # (n_points, 9)
    # rbf_values = np.exp(-effective_width * dist_sq)  # (n_points, 9)
    
    x_col = x[:, np.newaxis]
    centers_row = centers[np.newaxis, :]
    dist_sq = (x_col - centers_row)**2
    rbf_values = np.exp(-effective_width * dist_sq)
    
    # Prediction: sum of weights * rbf_values
    # weights has shape (9,), rbf_values has shape (n_points, 9)
    # pred = rbf_values @ weights  # (n_points,)
    pred = rbf_values @ weights
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 140:  # Safety margin before 150s limit
        if not np.all(np.isfinite(pred)):
            pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        return pred
    
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
# EVOLVE-BLOCK-END
