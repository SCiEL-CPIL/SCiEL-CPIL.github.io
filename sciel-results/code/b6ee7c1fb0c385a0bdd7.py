"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _spherical_circumcenters(points, facets):
    """
    Compute circumcenters of spherical triangles defined by points and facets.
    Returns an array of shape (n_facets, 3) with normalized circumcenter directions.
    """
    a = points[facets[:, 0]]
    b = points[facets[:, 1]]
    c = points[facets[:, 2]]
    
    ab = np.cross(a, b)
    bc = np.cross(b, c)
    ca = np.cross(c, a)
    
    n = ab + bc + ca
    norms_n = np.linalg.norm(n, axis=1, keepdims=True)
    
    valid = norms_n.squeeze() > 1e-12
    centers = np.zeros_like(a)
    
    if np.any(valid):
        centers[valid] = n[valid] / norms_n[valid]
        
    return centers

def _get_hull_data(points):
    """
    Compute convex hull and return hull, facets, centers, and per-facet circumradii in radians.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        if len(facets) == 0:
            return None, None, None, None
            
        centers = _spherical_circumcenters(points, facets)
        
        a = points[facets[:, 0]]
        cos_R = np.sum(a * centers, axis=1)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        
        R = np.arccos(cos_R)
        return hull, facets, centers, R
    except Exception:
        return None, None, None, None

def _covering_radius_fast(points):
    """
    Compute the covering radius in degrees.
    """
    _, _, _, R = _get_hull_data(points)
    if R is None:
        return np.inf
    return np.degrees(np.max(R))

def _get_worst_facet(points):
    """
    Return the index of the facet with maximum circumradius, and the circumradius in radians.
    """
    _, facets, centers, R = _get_hull_data(points)
    if R is None:
        return None, np.inf
    worst_idx = np.argmax(R)
    return worst_idx, R[worst_idx]

def _optimize_worst_facet(points, worst_idx, max_iter=20, tol=1e-8):
    """
    Optimize the 3 points of the worst facet to minimize its circumradius,
    while keeping other points fixed. Uses coordinate descent on the 3 points.
    """
    _, facets, centers, R = _get_hull_data(points)
    if R is None or worst_idx is None:
        return points, R[worst_idx] if R is not None else np.inf
    
    facet = facets[worst_idx]
    v0, v1, v2 = facet
    
    current_best = points.copy()
    current_best_R = R[worst_idx]
    
    for iteration in range(max_iter):
        improved = False
        
        # Try moving each of the 3 vertices
        for vi in range(3):
            orig_idx = facet[vi]
            orig_pt = points[orig_idx].copy()
            
            # Try small perturbations in random directions
            for _ in range(5):
                # Random direction on the sphere
                rand_dir = np.random.randn(3)
                rand_dir /= np.linalg.norm(rand_dir)
                
                # Move point slightly in this direction
                step_size = 0.01
                new_pt = points[orig_idx] + step_size * rand_dir
                new_pt /= np.linalg.norm(new_pt)
                
                # Update and evaluate
                test_points = points.copy()
                test_points[orig_idx] = new_pt
                
                _, _, _, test_R = _get_hull_data(test_points)
                if test_R is not None:
                    # Check if the worst facet's radius improved
                    # Find the facet that corresponds to the same 3 vertices
                    # (hull may have changed, so we need to be careful)
                    test_worst_R = np.max(test_R)
                    
                    if test_worst_R < current_best_R - tol:
                        current_best = test_points.copy()
                        current_best_R = test_worst_R
                        improved = True
        
        if not improved:
            break
    
    return current_best, current_best_R

def _lloyd_step(points):
    """
    Perform one iteration of Spherical Lloyd's Algorithm.
    Moves each point to the centroid of its Voronoi cell (approximated by avg of adjacent facet circumcenters).
    """
    try:
        hull, facets, centers, _ = _get_hull_data(points)
        if facets is None:
            return points
            
        new_points = np.zeros_like(points)
        counts = np.zeros(N)
        
        np.add.at(new_points, facets[:, 0], centers)
        np.add.at(new_points, facets[:, 1], centers)
        np.add.at(new_points, facets[:, 2], centers)
        
        np.add.at(counts, facets[:, 0], 1)
        np.add.at(counts, facets[:, 1], 1)
        np.add.at(counts, facets[:, 2], 1)
        
        valid = counts > 0
        new_points[valid] /= counts[valid, None]
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        valid_norms = norms.squeeze() > 1e-12
        new_points[valid_norms] /= norms[valid_norms]
        
        new_points[~valid] = points[~valid]
        new_points[~valid_norms] = points[~valid_norms]
        
        return new_points
    except Exception:
        return points

def _optimize_from_start(points, time_budget=10.0):
    """
    Run Lloyd iterations followed by worst-facet optimization.
    """
    start_time = time.time()
    current = points.copy()
    current_R = _covering_radius_fast(current)
    
    # Phase 1: Lloyd iterations
    for _ in range(30):
        if time.time() - start_time > time_budget * 0.6:
            break
        current = _lloyd_step(current)
        new_R = _covering_radius_fast(current)
        if new_R < current_R:
            current_R = new_R
        else:
            # Revert if it got worse
            current = _lloyd_step(current)  # Try once more
            new_R = _covering_radius_fast(current)
            if new_R >= current_R:
                break
    
    # Phase 2: Worst-facet optimization
    if time.time() - start_time < time_budget * 0.9:
        worst_idx, worst_R = _get_worst_facet(current)
        if worst_idx is not None:
            for _ in range(3):
                if time.time() - start_time > time_budget * 0.95:
                    break
                current, worst_R = _optimize_worst_facet(current, worst_idx, max_iter=10)
                new_R = _covering_radius_fast(current)
                if new_R < current_R:
                    current_R = new_R
    
    return current, current_R

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def fibonacci_spiral(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    best_points = fibonacci_spiral(N)
    best_radius = _covering_radius_fast(best_points)
    
    np.random.seed(42)
    
    # Multiple random restarts
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
            
        # Different initializations
        if restart == 0:
            init = best_points.copy()
        elif restart == 1:
            init = fibonacci_spiral(N, offset=0.1)
        elif restart == 2:
            init = fibonacci_spiral(N, offset=0.9)
        else:
            init = np.random.randn(N, 3)
            init /= np.linalg.norm(init, axis=1, keepdims=True)
        
        # Optimize from this start
        time_budget = min(15.0, deadline - 30.0 - (time.time() - start_time))
        if time_budget < 2.0:
            break
            
        candidate, candidate_R = _optimize_from_start(init, time_budget)
        
        if candidate_R < best_radius:
            best_radius = candidate_R
            best_points = candidate.copy()
    
    # Final polish with Nelder-Mead if time allows
    if time.time() < deadline - 10.0:
        best_flat = best_points.flatten()
        
        def objective(flat):
            pts = flat.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            valid = norms.squeeze() > 1e-12
            if not np.all(valid):
                return np.inf
            pts[valid] /= norms[valid]
            return _covering_radius_fast(pts)
            
        try:
            res = minimize(
                objective, 
                best_flat, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res.fun < best_radius:
                best_points = res.x.reshape(N, 3)
                best_radius = res.fun
                norms = np.linalg.norm(best_points, axis=1, keepdims=True)
                best_points /= norms
        except Exception:
            pass

    # Final safety normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points /= norms
    
    return best_points
# EVOLVE-BLOCK-END
