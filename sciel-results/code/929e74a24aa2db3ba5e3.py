"""Improved packing: hexagonal init + pairwise repulsion refinement with boundary handling."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(pts):
    """Compute the radius for a given set of points."""
    # Boundary distance
    bx = 0.5 - np.abs(pts[:, 0])
    by = 0.5 - np.abs(pts[:, 1])
    bound = np.min(np.minimum(bx, by))
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(bound, min_pair)


def _hex_init(n, margin=0.02):
    """Generate hexagonal lattice points in [-0.5+margin, 0.5-margin]^2."""
    # Hexagonal lattice: spacing s, rows offset
    # Area per point ~ s^2 * sqrt(3)/2
    # We want ~n points in area (1-2*margin)^2
    area = (1.0 - 2*margin)**2
    s = np.sqrt(2.0 * area / (n * np.sqrt(3)))
    
    # Generate hex grid
    row_height = s * np.sqrt(3) / 2
    x_spacing = s
    y_spacing = row_height
    
    pts = []
    # Determine how many rows and cols we need
    half = 0.5 - margin
    n_rows = int(np.ceil(2 * half / y_spacing)) + 2
    n_cols = int(np.ceil(2 * half / x_spacing)) + 2
    
    for row in range(-n_rows//2, n_rows//2 + 1):
        y = row * y_spacing
        if y > half or y < -half:
            continue
        offset = (x_spacing / 2) if (row % 2 != 0) else 0.0
        for col in range(-n_cols//2, n_cols//2 + 1):
            x = col * x_spacing + offset
            if x <= half and x >= -half:
                pts.append([x, y])
    
    pts = np.array(pts)
    
    # If we have more than n, randomly subsample
    if len(pts) > n:
        idx = np.random.choice(len(pts), n, replace=False)
        pts = pts[idx]
    elif len(pts) < n:
        # Pad with random points
        extra = np.random.uniform(-half, half, (n - len(pts), 2))
        pts = np.vstack([pts, extra])
    
    return pts


def _repulsion_step(pts, strength=1.0, n_iter=100, learning_rate=0.01):
    """Iteratively push points apart using repulsion forces."""
    n = len(pts)
    pts = pts.copy()
    
    for it in range(n_iter):
        # Compute pairwise distances and forces
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Repulsion force: 1/d^2 in direction of separation
        # Only consider close pairs for efficiency
        mask = dists < 0.15  # Only interact within this range
        forces = np.zeros_like(pts)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-10:
                    # Random nudge
                    forces[i] += np.random.randn(2) * 0.001
                    continue
                if d < 0.15:
                    # Repulsion
                    f = strength / (d * d)
                    forces[i] += f * (diff[i, j] / d)
        
        # Boundary forces: push back if too close to edge
        for i in range(n):
            for k in range(2):
                if pts[i, k] > 0.5 - 0.01:
                    forces[i, k] -= strength * (pts[i, k] - (0.5 - 0.01)) * 10
                elif pts[i, k] < -0.5 + 0.01:
                    forces[i, k] += strength * ((-0.5 + 0.01) - pts[i, k]) * 10
        
        # Update positions
        pts += learning_rate * forces
        
        # Clip to square
        pts = np.clip(pts, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return pts


def _local_optimize(pts, n_iter=500, lr_init=0.005, lr_decay=0.999):
    """Local optimization using gradient-based approach on the min-distance objective."""
    n = len(pts)
    pts = pts.copy()
    lr = lr_init
    
    for it in range(n_iter):
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Find minimum pair
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_dist = dists[min_idx]
        
        # Boundary distances
        bx = 0.5 - np.abs(pts[:, 0])
        by = 0.5 - np.abs(pts[:, 1])
        bound_dists = np.minimum(bx, by)
        min_bound_idx = np.argmin(bound_dists)
        min_bound = bound_dists[min_bound_idx]
        
        # Determine which constraint is active
        if min_dist / 2.0 < min_bound:
            # Pairwise constraint active
            i, j = min_idx
            direction = (pts[i] - pts[j]) / (min_dist + 1e-15)
            pts[i] += lr * direction
            pts[j] -= lr * direction
        else:
            # Boundary constraint active
            i = min_bound_idx
            # Push away from nearest boundary
            for k in range(2):
                if abs(pts[i, k]) > 0.49:
                    pts[i, k] -= lr * np.sign(pts[i, k])
        
        # Clip
        pts = np.clip(pts, -0.5 + 1e-9, 0.5 - 1e-9)
        
        lr *= lr_decay
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial hexagonal configuration
    np.random.seed(42)
    pts = _hex_init(N, margin=0.015)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Phase 1: Repulsion-based refinement
    while time.time() < deadline - 20.0:
        pts = _repulsion_step(pts, strength=0.5, n_iter=50, learning_rate=0.005)
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Phase 2: Local optimization
    pts = best_pts.copy()
    while time.time() < deadline - 5.0:
        pts = _local_optimize(pts, n_iter=200, lr_init=0.003, lr_decay=0.9995)
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Final polish
    pts = best_pts.copy()
    while time.time() < deadline - 1.0:
        pts = _local_optimize(pts, n_iter=100, lr_init=0.001, lr_decay=0.9999)
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
