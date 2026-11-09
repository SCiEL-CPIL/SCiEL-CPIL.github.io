"""
Strategy: Fixed Gaussian RBF basis with 10 learnable amplitudes.
This replaces the global polynomial basis with local basis functions.
BFGS optimizes only the linear weights, which is a convex problem (minimizing MSE with fixed basis),
leading to reliable convergence to the best fit within this basis family.
The fixed centers are chosen to cover the domain [0.5, 2.0]^3.
"""
import numpy as np
import time

# Fixed centers for the 10 RBFs in the domain [0.5, 2.0]^3
# We use a mix of corners, edge midpoints, and center to ensure good coverage.
# Domain size is 1.5 in each dimension.
# Let's define 10 centers:
# 8 corners of the cube [0.5, 2.0]^3
# 1 center (1.25, 1.25, 1.25)
# 1 additional point, maybe (1.25, 0.5, 0.5) to add asymmetry/coverage
centers = np.array([
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 2.0],
    [0.5, 2.0, 0.5],
    [0.5, 2.0, 2.0],
    [2.0, 0.5, 0.5],
    [2.0, 0.5, 2.0],
    [2.0, 2.0, 0.5],
    [2.0, 2.0, 2.0],
    [1.25, 1.25, 1.25],
    [1.25, 0.5, 0.5]
])

# Width of the Gaussians. 
# If sigma is too small, the basis functions are very local and might be linearly dependent or hard to fit.
# If sigma is too large, they become global and similar to a constant or low-order poly.
# A value around 0.6-0.8 seems reasonable for domain size 1.5.
SIGMA = 0.7
SIGMA_SQ_2 = 2.0 * SIGMA * SIGMA

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    # x, y, z are arrays of shape (N,)
    # params is an array of shape (10,)
    
    # Stack inputs into shape (N, 3)
    # Ensure inputs are float
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    N = len(x)
    
    # Compute distances from each point to each center
    # Points: (N, 3)
    # Centers: (10, 3)
    # We want dist_sq[i, j] = ||point_i - center_j||^2
    
    # Efficient computation:
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    # points_norm_sq: (N, 1)
    # centers_norm_sq: (1, 10)
    # dot_prod: (N, 10)
    
    points = np.stack((x, y, z), axis=-1) # (N, 3)
    
    points_norm_sq = np.sum(points * points, axis=1)[:, np.newaxis] # (N, 1)
    centers_norm_sq = np.sum(centers * centers, axis=1)[np.newaxis, :] # (1, 10)
    dot_prod = points @ centers.T # (N, 10)
    
    dist_sq = points_norm_sq + centers_norm_sq - 2.0 * dot_prod
    
    # Clip to avoid negative values due to floating point errors
    dist_sq = np.maximum(dist_sq, 0.0)
    
    # Compute Gaussian values: exp(-dist_sq / (2 * sigma^2))
    # (N, 10)
    basis_vals = np.exp(-dist_sq / SIGMA_SQ_2)
    
    # Prediction is linear combination: basis_vals @ params
    # (N, 10) @ (10,) -> (N,)
    pred = basis_vals @ p
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        # Fallback to a simple linear model if something goes wrong
        # This should rarely happen
        pred = np.ones_like(x)
        
    # Deadline check (not strictly necessary for this fast op, but good practice)
    elapsed = time.time() - start_time
    if elapsed > 110.0:
         # If this ever takes too long (it won't), we just return what we have.
         pass
         
    return pred
