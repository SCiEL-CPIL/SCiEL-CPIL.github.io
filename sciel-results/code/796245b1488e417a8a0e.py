"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _generate_points(r1, r2, theta1, theta2, dy):
    """Generate 8 points: two concentric regular squares with independent radii, orientations, and vertical offset."""
    # Square 1: radius r1, orientation theta1
    angles1 = theta1 + np.pi * np.array([0, 0.5, 1.0, 1.5])
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    
    # Square 2: radius r2, orientation theta2, vertical offset dy
    angles2 = theta2 + np.pi * np.array([0, 0.5, 1.0, 1.5])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2) + dy])
    
    return np.vstack([pts1, pts2])

def _score(params):
    """Compute negative log of the score (to minimize). Returns large value if invalid."""
    r1, r2, theta1, theta2, dy = params
    if r1 < 0 or r2 < 0:
        return 1e10
    
    pts = _generate_points(r1, r2, theta1, theta2, dy)
    
    # Check distinctness
    min_dist = np.inf
    for i in range(8):
        for j in range(i + 1, 8):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return 1e10
            min_dist = min(min_dist, d)
    
    # Compute diameter
    max_dist = 0.0
    for i in range(8):
        for j in range(i + 1, 8):
            d = np.linalg.norm(pts[i] - pts[j])
            max_dist = max(max_dist, d)
    
    if max_dist < 1e-12:
        return 1e10
    
    # Scale to diameter 2
    scale = 2.0 / max_dist
    pts_scaled = pts * scale
    
    # Compute product of distances over ordered pairs
    # Ordered pairs: i != j, so 8*7 = 56 pairs
    log_product = 0.0
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            d = np.linalg.norm(pts_scaled[i] - pts_scaled[j])
            if d < 1e-15:
                return 1e10
            log_product += np.log(d)
    
    # Score = Delta / n^n where Delta is product of distances
    # log(score) = log_product - n * log(n)
    n = 8
    log_score = log_product - n * np.log(n)
    
    # We want to maximize score, so minimize -log_score
    return -log_score

def construct():
    # Optimize the 5-parameter manifold
    # Bounds: r1, r2 in [0.1, 2.0], theta1, theta2 in [0, pi/2], dy in [-1, 1]
    bounds = [
        (0.1, 2.0),  # r1
        (0.1, 2.0),  # r2
        (0.0, np.pi / 2),  # theta1
        (0.0, np.pi / 2),  # theta2
        (-1.0, 1.0),  # dy
    ]
    
    # Initial guess: regular octagon corresponds to r1 = r2 = 1/sqrt(2) * sqrt(2) = 1? 
    # Actually, regular octagon vertices are at distance 1 from origin with angles k*pi/4
    # Two squares: one with vertices at angles 0, pi/2, pi, 3pi/2 (square 1)
    # and one with vertices at angles pi/4, 3pi/4, 5pi/4, 7pi/4 (square 2)
    # Both have radius 1
    # So r1 = r2 = 1, theta1 = 0, theta2 = pi/4, dy = 0
    
    result = differential_evolution(
        _score,
        bounds,
        seed=42,
        maxiter=1000,
        tol=1e-10,
        polish=True,
        workers=1
    )
    
    if result.success or result.fun < 1e9:
        best_params = result.x
    else:
        # Fallback to regular octagon
        best_params = [1.0, 1.0, 0.0, np.pi / 4, 0.0]
    
    pts = _generate_points(*best_params)
    
    # Scale to diameter 2
    max_dist = 0.0
    for i in range(8):
        for j in range(i + 1, 8):
            d = np.linalg.norm(pts[i] - pts[j])
            max_dist = max(max_dist, d)
    
    if max_dist < 1e-12:
        # Fallback to regular octagon
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
    
    scale = 2.0 / max_dist
    pts_scaled = pts * scale
    
    return [[float(pts_scaled[i][0]), float(pts_scaled[i][1])] for i in range(8)]
# EVOLVE-BLOCK-END
