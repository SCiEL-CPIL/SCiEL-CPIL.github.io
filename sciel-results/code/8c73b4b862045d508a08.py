"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _generate_points(r1, r2, theta1, theta2, dy):
    """
    Generate 8 points: two regular squares.
    Square 1: center (0,0), radius r1, orientation theta1.
    Square 2: center (0,dy), radius r2, orientation theta2.
    """
    # Square 1
    angles1 = theta1 + np.pi * np.array([0, 0.5, 1.0, 1.5])
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    
    # Square 2
    angles2 = theta2 + np.pi * np.array([0, 0.5, 1.0, 1.5])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2) + dy])
    
    return np.vstack([pts1, pts2])

def _score(params):
    """
    Compute negative log of the score (to minimize).
    Score = (Product of ordered pair distances after scaling to diam 2) / 8^8
    Maximize Score <=> Maximize log(Score) <=> Minimize -log(Score)
    """
    r1, r2, theta1, theta2, dy = params
    
    # Basic validity checks
    if r1 < 1e-6 or r2 < 1e-6:
        return 1e10
        
    pts = _generate_points(r1, r2, theta1, theta2, dy)
    
    # Compute pairwise distances
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Check for distinctness (diagonal is 0, off-diagonal should be > 0)
    # We only care about i != j
    mask = ~np.eye(8, dtype=bool)
    off_diag_dists = dists[mask]
    
    if np.any(off_diag_dists < 1e-9):
        return 1e10
        
    D = np.max(off_diag_dists)
    
    if D < 1e-9:
        return 1e10
        
    # log_score = sum(log(d_ij)) - 56 * log(D) + const
    # We ignore const.
    # Sum of logs of all 56 off-diagonal distances
    sum_log_dists = np.sum(np.log(off_diag_dists))
    
    # 56 pairs
    log_D_term = 56.0 * np.log(D)
    
    # Objective to minimize: - (sum_log_dists - log_D_term)
    obj = -(sum_log_dists - log_D_term)
    
    return obj

def construct():
    # Bounds for parameters
    # r1, r2: Radii. Regular octagon has r=1. 
    # If we scale to diameter 2, the absolute scale doesn't matter for the final score, 
    # but it matters for the optimization landscape. 
    # Regular octagon diameter is 2 (if r=1, dist between opposite vertices is 2).
    # So r=1 is a good scale. Let's allow [0.1, 5.0]
    
    # theta1, theta2: Orientations. Periodic with pi/2.
    # Regular octagon: one square at 0, one at pi/4.
    # Let's search [0, pi/2]
    
    # dy: Vertical offset. Regular octagon has 0.
    # Let's search [-2.0, 2.0] to allow significant separation.
    
    bounds = [
        (0.1, 5.0),  # r1
        (0.1, 5.0),  # r2
        (0.0, np.pi / 2),  # theta1
        (0.0, np.pi / 2),  # theta2
        (-2.0, 2.0)   # dy
    ]
    
    # Regular octagon parameters:
    # r1 = 1, r2 = 1, theta1 = 0, theta2 = pi/4, dy = 0
    initial_guess = [1.0, 1.0, 0.0, np.pi / 4, 0.0]
    
    # Run differential evolution
    # Using a fixed seed for reproducibility
    result = differential_evolution(
        _score,
        bounds,
        seed=42,
        maxiter=2000,
        tol=1e-10,
        polish=True,
        workers=1,
        init='sobol' # Sobol sequence for better coverage
    )
    
    if result.success or result.fun < 1e9:
        best_params = result.x
    else:
        best_params = initial_guess
        
    pts = _generate_points(*best_params)
    
    # Scale to diameter 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    mask = ~np.eye(8, dtype=bool)
    off_diag_dists = dists[mask]
    D = np.max(off_diag_dists)
    
    if D < 1e-9:
        # Fallback to regular octagon
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
        
    scale = 2.0 / D
    pts_scaled = pts * scale
    
    return [[float(pts_scaled[i][0]), float(pts_scaled[i][1])] for i in range(8)]
# EVOLVE-BLOCK-END
