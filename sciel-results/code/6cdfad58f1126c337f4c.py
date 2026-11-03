"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

def _compute_radius(points):
    """Compute the maximum common radius for given centers."""
    if len(points) < 2:
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.linalg.norm(points, axis=1)
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Minimum pairwise distance / 2
    # For efficiency with 300 points, we can use cdist
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    r_pair = min_pair / 2.0
    return min(r_boundary, r_pair)

def _hex_lattice_init(n):
    """Generate a hexagonal lattice initial configuration."""
    # Estimate spacing
    area_per_circle = np.pi / n
    spacing = np.sqrt(4 * area_per_circle / np.sqrt(3))
    
    points = []
    # Generate points in hexagonal grid
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0 - spacing/2.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n:
        # Fallback: random points
        theta = np.random.uniform(0, 2*np.pi, n)
        r = np.sqrt(np.random.uniform(0, 1, n))
        points = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
    else:
        # Select n points closest to center for better initial packing
        radii = np.linalg.norm(points, axis=1)
        idx = np.argsort(radii)[:n]
        points = points[idx]
    
    return points

def _local_optimize(points, max_iter=50, tol=1e-8):
    """
    Local optimization using gradient-free approach.
    We maximize the minimum of:
      - distance to boundary for each point
      - half the minimum pairwise distance
    
    This is equivalent to maximizing r where:
      r <= 1 - |p_i| for all i
      r <= |p_i - p_j|/2 for all i<j
    """
    points = points.copy()
    n = len(points)
    
    # We'll use a simple iterative improvement:
    # At each step, find the most constrained point and try to move it.
    
    best_r = _compute_radius(points)
    
    for iteration in range(max_iter):
        # Compute current radius
        dists_boundary = 1.0 - np.linalg.norm(points, axis=1)
        d = cdist(points, points)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        r = min(np.min(dists_boundary), min_pair_dist / 2.0)
        
        if r <= best_r:
            best_r = r
        
        # Find the point that is most "tight" (closest to violating constraint)
        # and try to nudge it.
        
        # For each point, compute how much "slack" it has
        # slack_i = min(dists_boundary[i], min_{j!=i} d[i,j]/2)
        slacks = dists_boundary.copy()
        for i in range(n):
            min_d_i = np.min(d[i])
            slacks[i] = min(slacks[i], min_d_i / 2.0)
        
        # Find the point with minimum slack
        i_min = np.argmin(slacks)
        
        # Try to move this point in a direction that increases its slack
        # The direction should be away from the nearest neighbor and/or away from boundary
        nearest_neighbor = np.argmin(d[i_min])
        if nearest_neighbor != i_min:
            # Direction away from nearest neighbor
            dir_away = points[i_min] - points[nearest_neighbor]
            dir_away_norm = np.linalg.norm(dir_away)
            if dir_away_norm > 1e-10:
                dir_away = dir_away / dir_away_norm
            else:
                dir_away = np.array([1.0, 0.0])
        
        # Also consider direction away from boundary (inward)
        pos_norm = np.linalg.norm(points[i_min])
        if pos_norm > 1e-10:
            dir_inward = -points[i_min] / pos_norm
        else:
            dir_inward = np.array([0.0, 0.0])
        
        # Combine directions
        dir_total = dir_away + 0.5 * dir_inward
        dir_total_norm = np.linalg.norm(dir_total)
        if dir_total_norm > 1e-10:
            dir_total = dir_total / dir_total_norm
        
        # Try small steps
        improved = False
        for step_size in [0.01, 0.005, 0.001, 0.0005]:
            new_point = points[i_min] + step_size * dir_total
            new_points = points.copy()
            new_points[i_min] = new_point
            
            # Check if still inside container
            if np.linalg.norm(new_point) >= 1.0:
                continue
            
            new_r = _compute_radius(new_points)
            if new_r > r + tol:
                points = new_points
                improved = True
                break
        
        if not improved:
            # Try random perturbations
            for _ in range(10):
                delta = np.random.normal(0, 0.01, 2)
                new_point = points[i_min] + delta
                if np.linalg.norm(new_point) >= 1.0:
                    continue
                new_points = points.copy()
                new_points[i_min] = new_point
                new_r = _compute_radius(new_points)
                if new_r > r + tol:
                    points = new_points
                    improved = True
                    break
        
        if not improved:
            break
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial configuration
    points = _hex_lattice_init(N)
    best_points = points.copy()
    best_r = _compute_radius(points)
    
    # Local optimization
    points = _local_optimize(points, max_iter=100)
    r = _compute_radius(points)
    if r > best_r:
        best_points = points.copy()
        best_r = r
    
    # Check time
    if time.time() > deadline:
        return best_points
    
    # Try multiple random restarts if time permits
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Random initialization with some structure
        theta = np.random.uniform(0, 2*np.pi, N)
        r_rand = np.sqrt(np.random.uniform(0.1, 0.9, N))
        points = np.column_stack([r_rand*np.cos(theta), r_rand*np.sin(theta)])
        
        # Ensure no overlaps in initial config (simple rejection)
        valid = True
        d = cdist(points, points)
        np.fill_diagonal(d, np.inf)
        if np.min(d) < 0.01:
            valid = False
        
        if not valid:
            continue
        
        points = _local_optimize(points, max_iter=50)
        r = _compute_radius(points)
        if r > best_r:
            best_points = points.copy()
            best_r = r
    
    # Final polish with more iterations
    if time.time() < deadline - 10:
        points = best_points.copy()
        points = _local_optimize(points, max_iter=200, tol=1e-10)
        r = _compute_radius(points)
        if r > best_r:
            best_points = points.copy()
            best_r = r
    
    return best_points
# EVOLVE-BLOCK-END
