# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

def _triangle_areas(points):
    """Compute areas of all C(n,3) triangles."""
    n = len(points)
    indices = list(range(n))
    areas = np.array([
        abs(points[i][0]*(points[j][1]-points[k][1]) + 
            points[j][0]*(points[k][1]-points[i][1]) + 
            points[k][0]*(points[i][1]-points[j][1])) / 2.0
        for i, j, k in combinations(indices, 3)
    ])
    return areas

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return np.inf

def _score(points):
    """Compute A_min / A_hull."""
    areas = _triangle_areas(points)
    a_min = np.min(areas)
    if a_min <= 0:
        return 0.0
    a_hull = _hull_area(points)
    if a_hull <= 0 or not np.isfinite(a_hull):
        return 0.0
    return a_min / a_hull

def _soft_min_penalty(points, tau):
    """Smooth approximation of -min(triangle_areas) using log-sum-exp."""
    areas = _triangle_areas(points)
    # Clip to avoid overflow
    areas_clipped = np.clip(areas, 1e-10, None)
    log_sum = np.log(np.sum(np.exp(-areas_clipped / tau)))
    return -tau * log_sum

def _objective(points_flat, tau):
    """Objective function for L-BFGS-B: minimize -soft_min_penalty / hull_area approximation."""
    points = points_flat.reshape(-1, 2)
    # We want to maximize min_area / hull_area
    # Approximate by maximizing soft_min_penalty (which approximates min_area)
    # and penalizing large hull area
    sm = _soft_min_penalty(points, tau)
    # Normalize hull area to keep it bounded
    a_hull = _hull_area(points)
    if not np.isfinite(a_hull) or a_hull <= 0:
        return 1e10
    # Objective: minimize -(sm / a_hull) = minimize -sm * (1/a_hull)
    # But hull area changes with points, so use a fixed reference
    # Simpler: just maximize sm (min area proxy), hull will be constrained by geometry
    return -sm

def _objective_with_hull(points_flat, tau, ref_hull):
    """Objective that accounts for hull area."""
    points = points_flat.reshape(-1, 2)
    sm = _soft_min_penalty(points, tau)
    a_hull = _hull_area(points)
    if not np.isfinite(a_hull) or a_hull <= 0:
        return 1e10
    return -(sm / a_hull)

def _coordinate_descent(points, max_iter=500, tol=1e-10):
    """Polish solution with coordinate descent."""
    points = points.copy()
    n = len(points)
    best_score = _score(points)
    
    for it in range(max_iter):
        improved = False
        for i in range(n):
            for coord in range(2):
                best_val = points[i][coord]
                best_local_score = best_score
                # Try small perturbations
                for delta in [0.01, 0.005, 0.001, -0.001, -0.005, -0.01]:
                    test_points = points.copy()
                    test_points[i][coord] += delta
                    s = _score(test_points)
                    if s > best_local_score + tol:
                        best_local_score = s
                        best_val = points[i][coord] + delta
                if best_val != points[i][coord]:
                    points[i][coord] = best_val
                    best_score = best_local_score
                    improved = True
        if not improved:
            break
    
    return points, best_score

def _hexagonal_lattice_13():
    """Create a hexagonal lattice arrangement for 13 points."""
    # Center + 6 inner ring + 6 outer ring
    points = [[0.0, 0.0]]
    # Inner ring (radius ~1)
    for i in range(6):
        angle = i * np.pi / 3
        points.append([np.cos(angle), np.sin(angle)])
    # Outer ring (radius ~2)
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        points.append([2 * np.cos(angle), 2 * np.sin(angle)])
    return np.array(points)

def _concentric_rings_13():
    """Create concentric ring arrangement for 13 points."""
    points = [[0.0, 0.0]]
    # Ring 1: 4 points
    for i in range(4):
        angle = i * np.pi / 2
        points.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    # Ring 2: 8 points
    for i in range(8):
        angle = i * np.pi / 4 + np.pi / 8
        points.append([1.8 * np.cos(angle), 1.8 * np.sin(angle)])
    return np.array(points)

def _spiral_13():
    """Create a spiral arrangement for 13 points."""
    points = []
    for i in range(13):
        if i == 0:
            points.append([0.0, 0.0])
        else:
            r = 0.3 * np.sqrt(i)
            theta = i * 2.4  # golden angle approximation
            points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points)

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area (Heilbronn problem).
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin
    
    # Generate diverse initializations
    inits = []
    
    # 1. Hexagonal lattice
    inits.append(_hexagonal_lattice_13())
    
    # 2. Concentric rings
    inits.append(_concentric_rings_13())
    
    # 3. Spiral
    inits.append(_spiral_13())
    
    # 4-7. Perturbed versions of hexagonal lattice
    rng = np.random.default_rng(42)
    for trial in range(4):
        base = _hexagonal_lattice_13()
        perturbed = base + rng.normal(0, 0.1, base.shape)
        inits.append(perturbed)
    
    # 8-10. Perturbed versions of concentric rings
    for trial in range(3):
        base = _concentric_rings_13()
        perturbed = base + rng.normal(0, 0.1, base.shape)
        inits.append(perturbed)
    
    # 11-13. Random points in unit square (normalized)
    for trial in range(3):
        random_pts = rng.random((13, 2))
        # Normalize to have similar scale
        random_pts = (random_pts - random_pts.mean(axis=0)) / (random_pts.std(axis=0) + 1e-10)
        inits.append(random_pts)
    
    best_points = None
    best_score = -np.inf
    
    # Try each initialization
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Normalize initial points to have reasonable scale
        pts = init_pts.copy()
        pts = (pts - pts.mean(axis=0)) / (np.linalg.norm(pts, axis=1).mean() + 1e-10)
        
        # L-BFGS-B optimization with decreasing tau
        current_pts = pts.copy().flatten()
        
        for tau in [0.1, 0.05, 0.02, 0.01, 0.005]:
            if time.time() > deadline:
                break
            
            try:
                result = minimize(
                    _objective,
                    current_pts,
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.success or result.fun < 1e8:
                    current_pts = result.x
            except Exception:
                pass
        
        # Convert back to points
        optimized_pts = current_pts.reshape(-1, 2)
        
        # Coordinate descent polishing
        if time.time() < deadline - 30:
            optimized_pts, cd_score = _coordinate_descent(optimized_pts, max_iter=200)
            if cd_score > best_score:
                best_score = cd_score
                best_points = optimized_pts
        else:
            s = _score(optimized_pts)
            if s > best_score:
                best_score = s
                best_points = optimized_pts
    
    # Final check
    if best_points is None:
        # Fallback to hexagonal lattice
        best_points = _hexagonal_lattice_13()
    
    # Ensure all points are finite
    if not np.all(np.isfinite(best_points)):
        best_points = _hexagonal_lattice_13()
    
    return best_points

# EVOLVE-BLOCK-END
