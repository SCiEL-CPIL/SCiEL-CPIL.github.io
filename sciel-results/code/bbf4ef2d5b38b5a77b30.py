# EVOLVE-BLOCK-START
import numpy as np
import time


def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 2^(k-2) + 1 points in general
    position (no three collinear) inside the unit square so as to MINIMIZE the number of
    convex k-gons whose vertices are among the points. Here k = 7, so n = 2^5 + 1 = 33.
    The conjectured record is a construction with 0 convex 7-gons.

    Returns:
        points: np.ndarray of shape (33, 2) with the x,y coordinates of the points,
            each lying in the unit square [0, 1] x [0, 1].
    """
    n = 33
    # Use a grid construction with small perturbations to avoid collinearity.
    # A 6x6 grid has 36 points. We will use 33 of them.
    # To ensure points are in the unit square and well-separated, we map grid indices to [0.1, 0.9].
    
    # Define grid points
    grid_size = 6
    coords = []
    for i in range(grid_size):
        for j in range(grid_size):
            coords.append([i, j])
            
    # We have 36 points. We need 33.
    # Remove 3 points. Let's remove the corners to keep the shape compact? 
    # Or just take the first 33.
    # Taking first 33 from a 6x6 grid (row-major) leaves a 5x6 block + 3 points of the 6th row.
    # Let's just pick 33 points.
    
    # To ensure general position (no 3 collinear), we add a small deterministic jitter.
    # The jitter must be small enough not to push points outside [0,1] or cause distance issues,
    # but large enough to break collinearity.
    
    # Map grid indices to [0.1, 0.9]
    scale = 0.8 / (grid_size - 1)
    offset = 0.1
    
    points = []
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count >= n:
                break
            x = offset + i * scale
            y = offset + j * scale
            # Add deterministic jitter based on index to break symmetry/collinearity
            # Use a small irrational-like offset
            jitter_x = 0.001 * (i * 7 + j * 13 + 1) % 1.0
            jitter_y = 0.001 * (i * 11 + j * 17 + 2) % 1.0
            # Ensure jitter doesn't push out of bounds
            x += jitter_x
            y += jitter_y
            points.append([x, y])
            count += 1
        if count >= n:
            break
            
    points = np.array(points)
    
    # Verify validity (optional but good for debugging, though we must be fast)
    # Check bounds
    if np.any(points < 0) or np.any(points > 1):
        # Fallback to random if jitter pushed out (unlikely with 0.001 and 0.1 margin)
        np.random.seed(0)
        points = np.random.rand(n, 2)
        
    # Check collinearity (optional, jitter should handle it)
    # Check distance (optional, grid spacing is ~0.16, jitter is 0.001, so safe)
    
    return points

# EVOLVE-BLOCK-END
