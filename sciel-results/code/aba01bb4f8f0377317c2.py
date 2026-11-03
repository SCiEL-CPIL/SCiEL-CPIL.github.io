"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _compute_radius(pts):
    """Compute the packing radius for a set of points in [-0.5, 0.5]^2."""
    if len(pts) < 2:
        return 0.0
    # Boundary distance
    dist_to_boundary = 0.5 - np.max(np.abs(pts), axis=1)
    r_boundary = np.min(dist_to_boundary)
    
    # Pairwise distances
    tree = cKDTree(pts)
    pairs = tree.query_pairs(1.0)  # only need pairs closer than some threshold for min
    if not pairs:
        r_pair = 0.5
    else:
        dmin = 1e9
        for i, j in pairs:
            d = np.linalg.norm(pts[i] - pts[j])
            if d < dmin:
                dmin = d
        r_pair = dmin / 2.0
    
    return min(r_boundary, r_pair)


def _hex_init(n):
    """Generate a hexagonal lattice initialization for n points in [-0.5, 0.5]^2."""
    # We want to pack n points. Start with a hexagonal grid and clip.
    # Radius of each circle in hex packing with spacing s is s/2.
    # Area per point in hex packing is (sqrt(3)/2) * s^2.
    # Total area = 1. So n * (sqrt(3)/2) * s^2 <= 1 => s^2 <= 2/(n*sqrt(3))
    # s <= sqrt(2/(n*sqrt(3)))
    
    # Let's estimate spacing
    s = np.sqrt(2.0 / (n * np.sqrt(3.0)))
    # Add a small margin to ensure we don't go out of bounds initially
    s *= 0.99
    
    pts = []
    # Generate hex grid points
    # y step is s * sqrt(3) / 2
    dy = s * np.sqrt(3.0) / 2.0
    dx = s
    
    # Center the grid
    # We need to cover [-0.5, 0.5] in both x and y
    # Number of rows
    n_rows = int(1.0 / dy) + 2
    y_start = -0.5 - dy
    y_end = 0.5 + dy
    
    for row in range(n_rows):
        y = y_start + row * dy
        if y < -0.5 or y > 0.5:
            continue
        # Offset for even/odd rows
        offset = 0.0 if row % 2 == 0 else dx / 2.0
        x_start = -0.5 - dx + offset
        x_end = 0.5 + dx + offset
        n_cols = int((x_end - x_start) / dx) + 1
        for col in range(n_cols):
            x = x_start + col * dx
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([x, y])
    
    pts = np.array(pts)
    
    # If we have more than n, randomly select n
    if len(pts) > n:
        indices = np.random.choice(len(pts), n, replace=False)
        pts = pts[indices]
    elif len(pts) < n:
        # Fallback to square grid if hex didn't generate enough (unlikely)
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        pts = grid[:n]
        
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with hexagonal lattice
    pts = _hex_init(N)
    current_radius = _compute_radius(pts)
    
    # Local search: bottleneck elimination
    # We will iteratively find the closest pair or boundary violation and move points to increase radius.
    
    # Parameters
    alpha = 0.001  # Step size
    n_iterations = 100000
    
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        # Compute current radius and identify bottleneck
        # For efficiency, we only need to check a few things.
        # 1. Boundary violation: find point with min dist to boundary
        dist_to_boundary = 0.5 - np.max(np.abs(pts), axis=1)
        min_boundary_idx = np.argmin(dist_to_boundary)
        min_boundary_val = dist_to_boundary[min_boundary_idx]
        
        # 2. Pairwise distance: find closest pair
        tree = cKDTree(pts)
        pairs = tree.query_pairs(2 * current_radius + 0.01)
        
        min_pair_dist = 1e9
        min_pair_i, min_pair_j = -1, -1
        for i, j in pairs:
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_pair_dist:
                min_pair_dist = d
                min_pair_i, min_pair_j = i, j
        
        min_pair_val = min_pair_dist / 2.0
        
        # Determine bottleneck
        if min_boundary_val < min_pair_val:
            # Boundary is bottleneck
            bottleneck_type = 'boundary'
            bottleneck_val = min_boundary_val
            bottleneck_idx = min_boundary_idx
        else:
            # Pair is bottleneck
            bottleneck_type = 'pair'
            bottleneck_val = min_pair_val
            bottleneck_idx = min_pair_i  # We'll handle both in pair case
        
        # Move points to alleviate bottleneck
        if bottleneck_type == 'boundary':
            # Move point away from boundary
            idx = bottleneck_idx
            p = pts[idx]
            # Direction away from boundary
            # Find which boundary is closest
            # If x > 0.5 - r, move left. If x < -0.5 + r, move right.
            # If y > 0.5 - r, move down. If y < -0.5 + r, move up.
            # Simplified: move towards center
            direction = -p / (np.linalg.norm(p) + 1e-12)
            # Check if this move violates other constraints
            new_p = p + alpha * direction
            # Clip to square
            new_p = np.clip(new_p, -0.5, 0.5)
            
            # Check if new position is valid (no overlap)
            # We can't easily check all pairs, so we just try it and revert if radius decreases
            old_p = pts[idx].copy()
            pts[idx] = new_p
            new_radius = _compute_radius(pts)
            if new_radius < current_radius:
                pts[idx] = old_p
            else:
                current_radius = new_radius
        else:
            # Pair is bottleneck
            i, j = min_pair_i, min_pair_j
            if i == -1:
                break
            p_i = pts[i]
            p_j = pts[j]
            # Move them apart
            direction = (p_i - p_j) / (np.linalg.norm(p_i - p_j) + 1e-12)
            new_p_i = p_i + alpha * direction / 2
            new_p_j = p_j - alpha * direction / 2
            
            # Clip to square
            new_p_i = np.clip(new_p_i, -0.5, 0.5)
            new_p_j = np.clip(new_p_j, -0.5, 0.5)
            
            old_p_i = p_i.copy()
            old_p_j = p_j.copy()
            pts[i] = new_p_i
            pts[j] = new_p_j
            new_radius = _compute_radius(pts)
            if new_radius < current_radius:
                pts[i] = old_p_i
                pts[j] = old_p_j
            else:
                current_radius = new_radius
        
        # Occasionally try a random perturbation to escape local minima
        if it % 1000 == 0 and it > 0:
            idx = np.random.randint(N)
            old_p = pts[idx].copy()
            pts[idx] += alpha * np.random.randn(2)
            pts[idx] = np.clip(pts[idx], -0.5, 0.5)
            new_radius = _compute_radius(pts)
            if new_radius < current_radius:
                pts[idx] = old_p
            else:
                current_radius = new_radius
    
    return pts
# EVOLVE-BLOCK-END
