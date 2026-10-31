"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import time

def _circle_in_square(x, y, r):
    """Check if circle (x, y, r) is inside unit square."""
    return (r <= x <= 1 - r) and (r <= y <= 1 - r) and r > 0

def _pairwise_distance(p1, p2):
    """Euclidean distance between two 2D points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def _voronoi_inradius(vor, point_index, points):
    """
    Compute the inradius of the Voronoi cell for point_index.
    The inradius is the minimum distance from the point to any edge of its Voronoi cell.
    """
    if point_index >= len(points):
        return 0.0
    
    # Get the Voronoi region for this point
    try:
        region = vor.regions[point_index]
    except (IndexError, AttributeError):
        return 0.0
    
    if len(region) == 0 or -1 in region:
        # Unbounded region - this shouldn't happen if points are well-placed
        # but if it does, use a small radius
        return 0.01
    
    point = points[point_index]
    min_dist = np.inf
    
    # For each edge in the Voronoi cell, compute distance from point to edge
    for i in range(len(region)):
        v1_idx = region[i]
        v2_idx = region[(i + 1) % len(region)]
        
        if v1_idx < 0 or v2_idx < 0:
            continue
        
        v1 = vor.vertices[v1_idx]
        v2 = vor.vertices[v2_idx]
        
        # Distance from point to line segment v1-v2
        # First, check if the perpendicular projection falls within the segment
        d = v2 - v1
        if np.linalg.norm(d) < 1e-12:
            dist = np.linalg.norm(point - v1)
        else:
            t = np.dot(point - v1, d) / np.dot(d, d)
            t = max(0, min(1, t))
            closest = v1 + t * d
            dist = np.linalg.norm(point - closest)
        
        min_dist = min(min_dist, dist)
    
    if min_dist == np.inf:
        return 0.01
    
    return min_dist

def _voronoi_packing_objective(params, N):
    """
    Objective function for Voronoi-based circle packing.
    params: array of shape (2*N,) with [x0, y0, x1, y1, ...]
    Returns: negative sum of inradii (we minimize negative sum = maximize sum)
    """
    points = params.reshape(-1, 2)
    
    # Check if all points are inside the unit square with some margin
    for i in range(N):
        x, y = points[i]
        if x < 0.01 or x > 0.99 or y < 0.01 or y > 0.99:
            return 1000.0 + np.sum(points**2)  # Large penalty
    
    # Compute Voronoi diagram
    try:
        vor = Voronoi(points)
    except Exception:
        return 1000.0
    
    # Compute inradii
    total_inradius = 0.0
    for i in range(N):
        r = _voronoi_inradius(vor, i, points)
        total_inradius += r
    
    # Also ensure circles don't overlap by checking pairwise distances
    # The Voronoi inradius should handle this, but let's add a penalty if circles overlap
    for i in range(N):
        ri = _voronoi_inradius(vor, i, points)
        for j in range(i + 1, N):
            rj = _voronoi_inradius(vor, j, points)
            dist = _pairwise_distance(points[i], points[j])
            if dist < ri + rj - 1e-6:
                overlap = ri + rj - dist
                total_inradius -= overlap * 10
    
    return -total_inradius

def _voronoi_packing_gradient(params, N):
    """
    Numerical gradient of the Voronoi packing objective.
    """
    eps = 1e-6
    grad = np.zeros_like(params)
    f0 = _voronoi_packing_objective(params, N)
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _voronoi_packing_objective(params_plus, N)
        grad[i] = (f_plus - f0) / eps
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Initial configuration: grid layout
    side = int(np.ceil(np.sqrt(N)))
    points = np.array([[(i + 0.5) / side, (j + 0.5) / side] for i in range(side) for j in range(side)][:N])
    
    # Flatten for optimization
    params = points.flatten()
    
    # Optimize using BFGS with gradient
    best_params = params.copy()
    best_obj = _voronoi_packing_objective(params, N)
    
    try:
        result = minimize(
            _voronoi_packing_objective,
            params,
            args=(N,),
            method='BFGS',
            jac=_voronoi_packing_gradient,
            options={'maxiter': 500, 'gtol': 1e-8}
        )
        
        if result.fun < best_obj:
            best_params = result.x
            best_obj = result.fun
        
        # Check deadline
        if time.time() > deadline:
            pass
        else:
            # Try a second round with different initial conditions
            # Perturb the initial grid
            np.random.seed(42)
            for restart in range(3):
                if time.time() > deadline:
                    break
                
                # Create a slightly perturbed grid
                side2 = int(np.ceil(np.sqrt(N)))
                points2 = np.array([[(i + 0.5) / side2 + np.random.uniform(-0.02, 0.02), 
                                     (j + 0.5) / side2 + np.random.uniform(-0.02, 0.02)] 
                                    for i in range(side2) for j in range(side2)][:N])
                # Clamp to [0.05, 0.95]
                points2 = np.clip(points2, 0.05, 0.95)
                params2 = points2.flatten()
                
                result2 = minimize(
                    _voronoi_packing_objective,
                    params2,
                    args=(N,),
                    method='BFGS',
                    jac=_voronoi_packing_gradient,
                    options={'maxiter': 300, 'gtol': 1e-8}
                )
                
                if result2.fun < best_obj:
                    best_params = result2.x
                    best_obj = result2.fun
                    
    except Exception as e:
        # If optimization fails, fall back to grid
        best_params = params
    
    # Extract best points
    best_points = best_params.reshape(-1, 2)
    
    # Compute inradii for the final configuration
    try:
        vor = Voronoi(best_points)
        radii = np.array([_voronoi_inradius(vor, i, best_points) for i in range(N)])
    except Exception:
        # Fallback: use grid radii
        side = int(np.ceil(np.sqrt(N)))
        radii = np.full(N, 0.5 / side - 1e-3)
    
    # Ensure circles are inside the unit square
    for i in range(N):
        x, y = best_points[i]
        r = radii[i]
        # Clamp radius so circle fits in square
        r = min(r, x, 1 - x, y, 1 - y)
        r = max(r, 1e-4)  # Minimum radius
        radii[i] = r
    
    # Final check: ensure no overlaps by reducing radii if needed
    for iter_pass in range(50):
        if time.time() > deadline:
            break
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dist = _pairwise_distance(best_points[i], best_points[j])
                if dist < radii[i] + radii[j]:
                    # Reduce both radii
                    excess = radii[i] + radii[j] - dist
                    radii[i] -= excess / 2
                    radii[j] -= excess / 2
                    radii[i] = max(radii[i], 1e-4)
                    radii[j] = max(radii[j], 1e-4)
                    changed = True
        if not changed:
            break
    
    # Build result array
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [best_points[i][0], best_points[i][1], radii[i]]
    
    return result
# EVOLVE-BLOCK-END
