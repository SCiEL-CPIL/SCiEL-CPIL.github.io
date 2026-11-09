# EVOLVE-BLOCK-START
import numpy as np
import time


def _check_feasibility(centers, sides, tol=1e-7):
    """Check if the configuration is feasible (all inside unit square, no overlaps)."""
    n = len(sides)
    # Check inside unit square
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
        if s <= 0:
            return False
    # Check no overlaps
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _objective(centers, sides):
    """Return sum of side lengths (the metric to maximize)."""
    return np.sum(sides)


def _local_optimize(centers, sides, max_iters=5000, step_size=0.01):
    """
    Local optimization using gradient-free hill climbing.
    We perturb centers and sides slightly, accepting improvements.
    """
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = _objective(best_centers, best_sides)
    
    if not _check_feasibility(best_centers, best_sides):
        return best_centers, best_sides
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        # Try a random perturbation
        dc = rng.normal(0, step_size, size=(n, 2))
        ds = rng.normal(0, step_size * 0.5, size=n)
        
        new_centers = best_centers + dc
        new_sides = best_sides + ds
        
        # Ensure sides are positive
        new_sides = np.maximum(new_sides, 1e-6)
        
        if not _check_feasibility(new_centers, new_sides):
            continue
        
        new_val = _objective(new_centers, new_sides)
        if new_val > best_val + 1e-12:
            best_centers = new_centers
            best_sides = new_sides
            best_val = new_val
    
    return best_centers, best_sides


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].

    Uses a grid baseline followed by local optimization to improve the sum.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    n = 10
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    
    # Local optimization
    centers, sides = _local_optimize(centers, sides, max_iters=20000, step_size=0.02)
    
    # Verify feasibility
    if not _check_feasibility(centers, sides):
        # Fall back to grid if optimization broke feasibility
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
    
    return centers, sides


# EVOLVE-BLOCK-END
