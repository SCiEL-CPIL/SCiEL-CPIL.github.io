"""Improved packer for 200 equal circles in a unit square."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 200
SIDE = 1.0
HALF = 0.5
DEADLINE = 135.0  # safety margin before 150s kill

def _score(points):
    """Compute the max common radius for given centers in [-0.5,0.5]^2."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check bounds
    if np.any(points[:, 0] < -HALF - 1e-12) or np.any(points[:, 0] > HALF + 1e-12):
        return 0.0
    if np.any(points[:, 1] < -HALF - 1e-12) or np.any(points[:, 1] > HALF + 1e-12):
        return 0.0
    # Distance to walls
    wall_dist = np.minimum(np.minimum(HALF - points[:, 0], points[:, 0] + HALF),
                           np.minimum(HALF - points[:, 1], points[:, 1] + HALF))
    min_wall = np.min(wall_dist)
    # Pairwise distances
    tree = cKDTree(points)
    pairs = tree.query_pairs(2.0)  # any distance < 2.0, which covers all
    if len(pairs) == 0:
        min_pair = np.inf
    else:
        d = np.sqrt(np.sum((points[pairs[:, 0]] - points[pairs[:, 1]])**2, axis=1))
        min_pair = np.min(d)
    return min(min_wall, min_pair / 2.0)

def _square_grid(n):
    """Generate a square grid of n points in [-0.5, 0.5]^2."""
    m = int(np.ceil(np.sqrt(n)))
    # Use m x m grid, take first n
    xs = (np.arange(m) + 0.5) / m - 0.5
    ys = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ys for x in xs])
    return grid[:n]

def _hex_grid(n):
    """Generate a hexagonal lattice of n points in [-0.5, 0.5]^2."""
    # Hexagonal packing: rows offset by half spacing
    # We want to fit n points. Try different row counts.
    best = None
    best_score = -1
    for rows in range(10, 20):
        cols = int(np.ceil(n / rows))
        # Spacing in x
        dx = 1.0 / cols
        dy = dx * np.sqrt(3) / 2
        # Total height
        total_h = (rows - 1) * dy
        if total_h > 1.0:
            # Scale down
            dy = 1.0 / (rows - 1)
            dx = dy * 2 / np.sqrt(3)
            if cols * dx > 1.0:
                continue
        points = []
        for r in range(rows):
            y = -0.5 + r * dy
            offset = 0.0 if r % 2 == 0 else dx / 2
            for c in range(cols):
                x = -0.5 + c * dx + offset
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append((x, y))
        points = np.array(points[:n])
        if len(points) == n:
            s = _score(points)
            if s > best_score:
                best_score = s
                best = points
    return best if best is not None else _square_grid(n)

def _refine(points, max_iters=500, lr=0.001, deadline=None):
    """Simple gradient-free local refinement using random perturbations."""
    best = points.copy()
    best_score = _score(best)
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        # Pick a random point and perturb it
        idx = np.random.randint(len(best))
        orig = best[idx].copy()
        step = lr * np.exp(-it / 100.0)
        best[idx] += np.random.randn(2) * step
        # Clamp to bounds
        best[idx, 0] = np.clip(best[idx, 0], -HALF, HALF)
        best[idx, 1] = np.clip(best[idx, 1], -HALF, HALF)
        s = _score(best)
        if s > best_score:
            best_score = s
        else:
            best[idx] = orig
    return best, best_score

def construct():
    start_time = time.time()
    
    # Start with hex grid as initial guess
    points = _hex_grid(N)
    if points is None:
        points = _square_grid(N)
    
    best = points.copy()
    best_score = _score(best)
    
    # Try several random restarts with refinement
    np.random.seed(42)
    for restart in range(20):
        if time.time() > DEADLINE:
            break
        
        # Create a perturbed version of the best solution
        pts = best + np.random.randn(N, 2) * 0.01
        pts[:, 0] = np.clip(pts[:, 0], -HALF, HALF)
        pts[:, 1] = np.clip(pts[:, 1], -HALF, HALF)
        
        refined, score = _refine(pts, max_iters=300, lr=0.002, deadline=DEADLINE)
        if score > best_score:
            best_score = score
            best = refined
    
    # Final refinement on the best
    if time.time() < DEADLINE - 10:
        refined, score = _refine(best, max_iters=500, lr=0.001, deadline=DEADLINE)
        if score > best_score:
            best_score = score
            best = refined
    
    return best
