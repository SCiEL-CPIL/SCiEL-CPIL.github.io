"""Packing 200 equal circles in a unit square via hexagonal lattice + local optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 200
DEADLINE = 110.0  # seconds, safety margin before 150s kill

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    # Distance from each center to the nearest wall
    walls = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    wall_min = np.min(walls)
    
    # Minimum pairwise distance / 2
    # Use scipy's cKDTree for efficiency
    from scipy.spatial import cKDTree
    tree = cKDTree(centers)
    # Query nearest neighbor (excluding self)
    dists, _ = tree.query(centers, k=2)
    pair_min = np.min(dists[:, 1]) / 2.0
    
    return min(wall_min, pair_min)

def _hex_init(n):
    """Generate initial positions using a hexagonal lattice pattern."""
    # For n points in a hexagonal arrangement
    # Try different orientations and spacings
    best = None
    best_r = -1
    
    for orientation in [0, np.pi/6, np.pi/3]:
        # Generate hexagonal lattice points
        # The hexagonal lattice has spacing s in both x and y directions (approximately)
        # For n points, we need roughly sqrt(n) x sqrt(n) grid with hex offset
        
        m = int(np.ceil(np.sqrt(n)))
        # Hexagonal lattice: rows offset by half spacing
        # Spacing in x: s, spacing in y: s * sqrt(3)/2
        # Try to fit m x m points
        
        for scale in np.linspace(0.8, 1.2, 50):
            s = scale / m
            dx = s
            dy = s * np.sqrt(3) / 2
            
            pts = []
            for i in range(m):
                for j in range(m):
                    x = (j - (m-1)/2) * dx
                    y = (i - (m-1)/2) * dy
                    # Offset odd rows
                    if i % 2 == 1:
                        x += dx / 2
                    pts.append([x, y])
            
            pts = np.array(pts[:n])
            
            # Rotate
            if orientation != 0:
                c, s_rot = np.cos(orientation), np.sin(orientation)
                R = np.array([[c, -s_rot], [s_rot, c]])
                pts = pts @ R.T
            
            # Scale to fit in square
            max_abs = np.max(np.abs(pts))
            if max_abs > 0.45:
                pts = pts * (0.45 / max_abs)
            
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best = pts.copy()
    
    return best if best is not None else _square_init(n)

def _square_init(n):
    """Fallback square grid initialization."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]

def _local_optimize(centers, time_limit=30.0):
    """Local optimization using gradient-free methods to maximize minimum radius."""
    start_time = time.time()
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    n = len(centers)
    
    # Use Nelder-Mead on the negative of the minimum radius
    # But this is high-dimensional (400 variables), so use a smarter approach
    # Instead, do iterative local moves
    
    # Strategy: repeatedly find the "bottleneck" constraint and adjust
    # 1. Find the minimum wall distance and minimum pair distance
    # 2. Move points to increase the minimum
    
    from scipy.spatial import cKDTree
    
    step_size = 0.01
    n_iter = 0
    
    while time.time() - start_time < time_limit:
        # Compute current radius components
        walls = np.minimum(
            np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
            np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
        )
        wall_min = np.min(walls)
        wall_argmin = np.argmin(walls)
        
        tree = cKDTree(centers)
        dists, idx = tree.query(centers, k=2)
        pair_min = np.min(dists[:, 1]) / 2.0
        pair_argmin = np.argmin(dists[:, 1])
        
        current_r = min(wall_min, pair_min)
        
        if current_r > best_radius + 1e-12:
            best_radius = current_r
            best_centers = centers.copy()
        
        # Determine which constraint is binding
        if wall_min < pair_min:
            # Move the point closest to wall away from wall
            i = wall_argmin
            # Direction: from wall toward center
            dx = 0.5 - abs(centers[i, 0])
            dy = 0.5 - abs(centers[i, 1])
            # Determine which wall
            if centers[i, 0] > 0:
                dir_x = -1
            else:
                dir_x = 1
            if centers[i, 1] > 0:
                dir_y = -1
            else:
                dir_y = 1
            
            # Move toward center
            norm = np.sqrt(dir_x**2 + dir_y**2)
            if norm > 0:
                centers[i, 0] += step_size * dir_x / norm
                centers[i, 1] += step_size * dir_y / norm
        else:
            # Move the pair with minimum distance apart
            i = pair_argmin
            j = idx[i, 1]
            # Move i away from j
            diff = centers[i] - centers[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                centers[i] += step_size * diff / norm
                centers[j] -= step_size * diff / norm
        
        # Clip to square
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        n_iter += 1
        
        # Adaptive step size
        if n_iter % 100 == 0:
            step_size *= 0.95
            if step_size < 1e-5:
                break
    
    # Final polish: try to increase radius by moving all points slightly
    # Use a simple gradient ascent on a smoothed objective
    
    return best_centers, best_radius

def _smooth_max_radius(centers):
    """Compute a smoothed version of the radius using log-sum-exp for soft-min."""
    walls = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    
    from scipy.spatial import cKDTree
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    # Soft min using log-sum-exp with small temperature
    beta = 50.0
    soft_wall = -np.log(np.sum(np.exp(-beta * walls))) / beta
    soft_pair = -np.log(np.sum(np.exp(-beta * pair_dists))) / beta
    
    # Soft min of the two
    return min(soft_wall, soft_pair)

def construct():
    start_time = time.time()
    
    # Initial position: hexagonal lattice
    centers = _hex_init(N)
    
    # Local optimization
    centers, r = _local_optimize(centers, time_limit=60.0)
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    if remaining > 10:
        # Try a few more random perturbations and re-optimize
        best_centers = centers.copy()
        best_r = r
        
        for restart in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Perturb
            perturbed = centers + np.random.normal(0, 0.01, (N, 2))
            perturbed[:, 0] = np.clip(perturbed[:, 0], -0.49, 0.49)
            perturbed[:, 1] = np.clip(perturbed[:, 1], -0.49, 0.49)
            
            perturbed, r2 = _local_optimize(perturbed, time_limit=min(remaining - 5, 15.0))
            
            if r2 > best_r:
                best_r = r2
                best_centers = perturbed.copy()
        
        centers = best_centers
    
    # Final validation and clipping
    centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
    centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
    
    return centers
