# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from scipy.optimize import minimize
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 575.0  # Safety margin before 630s limit

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    indices = np.array(list(itertools.combinations(range(n), 3)))
    
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Score = A_min / A_hull
    """
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    
    if a_hull <= 1e-10 or a_min <= 1e-10:
        return 0.0
    
    return a_min / a_hull

def _lloyd_relax(points, n_outer, iterations=200, tol=1e-10):
    """
    Lloyd's algorithm: move each interior point to the centroid of its Voronoi cell.
    Hull vertices are fixed. This enforces uniform density which helps maximize min triangle area.
    """
    points = points.copy()
    n = len(points)
    
    for it in range(iterations):
        if time.time() > _DEADLINE - 10:
            break
            
        # Compute Voronoi diagram
        try:
            vor = Voronoi(points)
        except Exception:
            break
        
        # For each interior point (indices n_outer to n-1), find its Voronoi cell
        # and compute the centroid of the bounded region
        for i in range(n_outer, n):
            # Find the Voronoi region for point i
            region_index = vor.point_region[i]
            region = vor.regions[region_index]
            
            if -1 in region or len(region) == 0:
                continue
                
            # Get the vertices of this Voronoi cell
            verts = vor.vertices[region]
            
            # Check if all vertices are finite (bounded cell)
            if not np.all(np.isfinite(verts)):
                continue
            
            # Compute centroid of the Voronoi cell
            if len(verts) >= 3:
                centroid = np.mean(verts, axis=0)
                
                # Only update if the new position is still inside the convex hull
                # (simple check: distance from center should be reasonable)
                center = np.mean(points[:n_outer], axis=0)
                if np.linalg.norm(centroid - center) < np.linalg.norm(points[i] - center) * 1.5 + 0.1:
                    points[i] = centroid
        
        # Check convergence
        if it > 10:
            # Simple convergence check: if points haven't moved much
            pass  # Could add more sophisticated convergence check
    
    return points

def _make_initial_guess_lloyd(rng, n_outer=7):
    """
    Create initial guess with regular polygon hull vertices and randomly placed interior points,
    then apply Lloyd's algorithm to relax interior points to Voronoi centroids.
    """
    n = 13
    n_inner = n - n_outer
    
    # Regular polygon vertices on unit circle
    angles = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Random interior points within the unit disk
    inner_angles = rng.random(n_inner) * 2 * np.pi
    inner_radii = rng.random(n_inner) * 0.7 + 0.1
    inner = np.column_stack([inner_radii * np.cos(inner_angles), inner_radii * np.sin(inner_angles)])
    
    points = np.vstack([outer, inner])
    
    # Apply Lloyd's algorithm to relax interior points
    points = _lloyd_relax(points, n_outer, iterations=100)
    
    return points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the area of the
    smallest triangle formed by these points relative to the convex hull area.
    
    Uses Lloyd's algorithm (Voronoi centroid relaxation) for interior points
    combined with optimization of hull vertices.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(seed=42)
    
    # Phase 1: Try multiple initial configurations with Lloyd relaxation
    num_starts = 20
    
    for i in range(num_starts):
        if time.time() > _DEADLINE - 30:
            break
        
        # Create structured initial guess with Lloyd relaxation
        x0 = _make_initial_guess_lloyd(rng).flatten()
        
        # Evaluate initial score
        initial_score = _score(x0.reshape(n, 2))
        
        if initial_score > best_score:
            best_score = initial_score
            best_points = x0.reshape(n, 2)
        
        # Powell optimization from this starting point
        try:
            res = minimize(
                lambda x: -_score(x.reshape(n, 2)), 
                x0, 
                method='Powell', 
                options={
                    'maxiter': 3000, 
                    'xtol': 1e-10, 
                    'ftol': 1e-12,
                    'niter': 80
                }
            )
            
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_points = res.x.reshape(n, 2)
                
        except Exception:
            continue
    
    # Phase 2: Further Lloyd relaxation on the best configuration
    if best_points is not None and time.time() < _DEADLINE - 60:
        # Determine how many points are on the hull
        try:
            hull = ConvexHull(best_points)
            n_hull = len(hull.vertices)
        except Exception:
            n_hull = 7
        
        # Apply Lloyd relaxation with the current hull structure
        relaxed = _lloyd_relax(best_points, n_hull, iterations=200)
        relaxed_score = _score(relaxed)
        
        if relaxed_score > best_score:
            best_score = relaxed_score
            best_points = relaxed
    
    # Phase 3: Final polish with Powell
    if best_points is not None and time.time() < _DEADLINE - 30:
        try:
            res3 = minimize(
                lambda x: -_score(x.reshape(n, 2)), 
                best_points.flatten(), 
                method='Powell', 
                options={
                    'maxiter': 5000, 
                    'xtol': 1e-12, 
                    'ftol': 1e-14,
                    'niter': 100
                }
            )
            final_score = -res3.fun
            if final_score > best_score:
                best_score = final_score
                best_points = res3.x.reshape(n, 2)
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: regular 7-gon with 6 interior points in a smaller 6-gon
        n_outer = 7
        n_inner = 6
        angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
        outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + np.pi/n_inner
        inner = 0.5 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
        best_points = np.vstack([outer, inner])
        
    return best_points

# EVOLVE-BLOCK-END
