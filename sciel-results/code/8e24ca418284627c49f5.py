"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use 10 fixed RBF (Gaussian) basis functions centered at evenly spaced points
    in [0.5, 2]^3. The linear combination of these basis functions, weighted by
    params[0:10], approximates an arbitrary smooth function.
    
    We place centers on a 2x2x2 grid plus 2 extra to make 10 total, or simply
    use 10 evenly spaced points along a space-filling curve or grid.
    For simplicity and coverage, use a 2x2x2 grid (8 points) plus 2 face centers.
    Actually, let's use a 2x2x2 grid (8 points) and then add 2 more points to make 10.
    Or better: use 10 points from a Latin hypercube or just spread them out.
    
    Let's use a 2x2x2 grid = 8 points, plus the center of the domain, plus one more.
    Actually, simplest: use 10 points from a regular grid. 3x3x3 would be 27, too many.
    Let's use 2x2x2 = 8, plus 2 additional points.
    
    Centers:
    Corners of [0.5, 2]^3:
    (0.5, 0.5, 0.5), (0.5, 0.5, 2.0), (0.5, 2.0, 0.5), (0.5, 2.0, 2.0)
    (2.0, 0.5, 0.5), (2.0, 0.5, 2.0), (2.0, 2.0, 0.5), (2.0, 2.0, 2.0)
    Plus:
    (1.25, 1.25, 1.25)  # center
    (1.25, 1.25, 0.5)   # extra
    
    Bandwidth gamma: chosen so that RBFs have reasonable support.
    Domain size ~1.5 in each dimension. A good gamma might be around 2-5.
    Let's use gamma = 3.0.
    """
    # Fixed centers
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
        [1.25, 1.25, 0.5]
    ])
    
    gamma = 3.0
    
    # Stack input points: (n, 3)
    X = np.column_stack((x, y, z))
    
    # Compute distances to each center: (n, 10)
    # X: (n, 3), centers: (10, 3)
    # diff: (n, 10, 3)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)  # (n, 10)
    
    # RBF values: (n, 10)
    rbf_vals = np.exp(-gamma * sq_dist)
    
    # Linear combination with params[0:10]
    # params has length 10, use all 10
    result = rbf_vals @ params[:10]
    
    return result
# EVOLVE-BLOCK-END
