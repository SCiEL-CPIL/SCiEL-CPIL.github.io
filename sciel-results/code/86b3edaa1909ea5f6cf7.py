# EVOLVE-BLOCK-START
import numpy as np
import time


def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 33 points
    in general position inside the unit square so as to MINIMIZE the number of
    convex 7-gons whose vertices are among the points.
    
    Uses the Erdos-Szekeres recursive construction with nested convex chains.
    Points are arranged in levels of sizes 1, 2, 4, 8, 16, 2, each forming
    a convex chain nested inside the previous level.
    """
    n = 33
    
    points = []
    
    # Level 0: 1 point at top
    points.append((0.5, 0.95))
    
    # Level 1: 2 points
    points.append((0.3, 0.8))
    points.append((0.7, 0.8))
    
    # Level 2: 4 points
    for i in range(4):
        x = 0.15 + 0.7 * i / 3
        points.append((x, 0.6))
    
    # Level 3: 8 points
    for i in range(8):
        x = 0.1 + 0.8 * i / 7
        points.append((x, 0.4))
    
    # Level 4: 16 points
    for i in range(16):
        x = 0.05 + 0.9 * i / 15
        points.append((x, 0.2))
    
    # Level 5: 2 points at bottom
    points.append((0.2, 0.05))
    points.append((0.8, 0.05))
    
    points = np.array(points)
    
    # Add small perturbations to avoid collinearity
    np.random.seed(42)
    perturb = np.random.uniform(-0.005, 0.005, size=(33, 2))
    points = points + perturb
    
    # Ensure points are in [0,1] x [0,1]
    points = np.clip(points, 0.001, 0.999)
    
    return points


# EVOLVE-BLOCK-END
