"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize

def _normalize(pts):
    """Normalize rows of pts to unit length."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(norms, 1e-15)

def _compute_covering_radius(points):
    """Compute the covering radius (in degrees) using convex hull facets."""
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
    except Exception:
        try:
            delaunay = Delaunay(points)
            facets = delaunay.simplices
        except Exception:
            return 90.0
    
    max_radius = 0.0
    for simplex in facets:
        facet_pts = points[simplex]
        if len(simplex) == 3:
            v1, v2, v3 = facet_pts
            circum = np.cross(v1, v2) + np.cross(v2, v3) + np.cross(v3, v1)
            norm = np.linalg.norm(circum)
            if norm < 1e-15:
                continue
            circum = circum / norm
            cos_dists = np.clip(np.abs(np.dot(circum, facet_pts)), -1.0, 1.0)
            ang_dists = np.arccos(cos_dists)
            facet_radius = np.max(ang_dists)
        elif len(simplex) == 2:
            midpoint = (facet_pts[0] + facet_pts[1])
            norm = np.linalg.norm(midpoint)
            if norm < 1e-15:
                continue
            midpoint = midpoint / norm
            cos_dists = np.clip(np.abs(np.dot(midpoint, facet_pts)), -1.0, 1.0)
            ang_dists = np.arccos(cos_dists)
            facet_radius = np.max(ang_dists)
        else:
            continue
        max_radius = max(max_radius, facet_radius)
    
    return np.degrees(max_radius)

def _lloyd_relaxation(points, n_iter=100, tol=1e-10):
    """Lloyd relaxation: move each point toward the centroid of its Voronoi cell."""
    points = np.array(points, dtype=float).copy()
    n = len(points)
    
    for iteration in range(n_iter):
        try:
            delaunay = Delaunay(points)
        except Exception:
            break
        
        new_points = np.zeros_like(points)
        
        for i in range(n):
            cell_circumcenters = []
            for simplex in delaunay.simplices:
                if i in simplex:
                    facet_pts = points[simplex]
                    if len(facet_pts) == 3:
                        v1, v2, v3 = facet_pts
                        circum = np.cross(v1, v2) + np.cross(v2, v3) + np.cross(v3, v1)
                        norm = np.linalg.norm(circum)
                        if norm > 1e-15:
                            circum = circum / norm
                            cell_circumcenters.append(circum)
                    elif len(facet_pts) == 2:
                        midpoint = (facet_pts[0] + facet_pts[1])
                        norm = np.linalg.norm(midpoint)
                        if norm > 1e-15:
                            midpoint = midpoint / norm
                            cell_circumcenters.append(midpoint)
            
            if len(cell_circumcenters) > 0:
                avg = np.mean(cell_circumcenters, axis=0)
                norm = np.linalg.norm(avg)
                if norm > 1e-15:
                    new_points[i] = avg / norm
                else:
                    new_points[i] = points[i]
            else:
                new_points[i] = points[i]
        
        new_points = _normalize(new_points)
        diff = np.max(np.linalg.norm(new_points - points, axis=1))
        points = new_points
        
        if diff < tol:
            break
    
    return points

def _facet_local_search(points, max_iter=200, time_budget=30.0):
    """Direct facet-based local search: move vertices of worst facets toward circumcenters."""
    start_time = time.time()
    points = np.array(points, dtype=float).copy()
    n = len(points)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
        except Exception:
            break
        
        # Compute radius for each facet
        facet_radii = []
        facet_circumcenters = []
        for simplex in facets:
            facet_pts = points[simplex]
            if len(simplex) == 3:
                v1, v2, v3 = facet_pts
                circum = np.cross(v1, v2) + np.cross(v2, v3) + np.cross(v3, v1)
                norm = np.linalg.norm(circum)
                if norm < 1e-15:
                    facet_radii.append(0.0)
                    facet_circumcenters.append(None)
                    continue
                circum = circum / norm
                cos_dists = np.clip(np.abs(np.dot(circum, facet_pts)), -1.0, 1.0)
                ang_dists = np.arccos(cos_dists)
                facet_radii.append(np.max(ang_dists))
                facet_circumcenters.append(circum)
            elif len(simplex) == 2:
                midpoint = (facet_pts[0] + facet_pts[1])
                norm = np.linalg.norm(midpoint)
                if norm < 1e-15:
                    facet_radii.append(0.0)
                    facet_circumcenters.append(None)
                    continue
                midpoint = midpoint / norm
                cos_dists = np.clip(np.abs(np.dot(midpoint, facet_pts)), -1.0, 1.0)
                ang_dists = np.arccos(cos_dists)
                facet_radii.append(np.max(ang_dists))
                facet_circumcenters.append(midpoint)
            else:
                facet_radii.append(0.0)
                facet_circumcenters.append(None)
        
        if not facet_radii:
            break
        
        max_radius = max(facet_radii)
        if max_radius < 1e-12:
            break
        
        # Find facets with radius close to max (within 1% of max)
        threshold = max_radius * 0.99
        worst_facet_indices = [i for i, r in enumerate(facet_radii) if r >= threshold]
        
        if not worst_facet_indices:
            break
        
        # Adaptive step size based on current radius
        step_size = min(0.1, max_radius * 0.5)
        
        # Move vertices of worst facets toward their circumcenters
        moved = False
        for fi in worst_facet_indices:
            simplex = facets[fi]
            circum = facet_circumcenters[fi]
            if circum is None:
                continue
            
            for idx in simplex:
                # Move point idx toward circumcenter
                current = points[idx]
                # Spherical interpolation step
                angle = np.arccos(np.clip(np.dot(current, circum), -1.0, 1.0))
                if angle < 1e-10:
                    continue
                # Move a fraction of the way
                new_pt = current * np.cos(step_size) + circum * np.sin(step_size)
                new_pt = new_pt / np.linalg.norm(new_pt)
                points[idx] = new_pt
                moved = True
        
        if not moved:
            break
        
        points = _normalize(points)
    
    return points

def _soft_max_objective(points, beta=500.0):
    """Compute a smooth surrogate for the covering radius using soft-max."""
    n_samples = 2000
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    cos_dists = samples @ points.T
    cos_dists = np.clip(cos_dists, -1.0, 1.0)
    max_cos = np.max(cos_dists, axis=1)
    angles = np.arccos(max_cos)
    
    max_angle = np.max(angles)
    shifted = angles - max_angle
    log_sum_exp = np.log(np.sum(np.exp(beta * shifted))) + beta * max_angle
    return log_sum_exp / beta

def _generate_initial_configurations(n_configs=10):
    """Generate multiple initial configurations for diversity."""
    configs = []
    
    for cfg_idx in range(n_configs):
        if cfg_idx == 0:
            # Golden-angle spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif cfg_idx == 1:
            # Random points
            rng = np.random.RandomState(42 + cfg_idx)
            points = rng.randn(N, 3)
            points = _normalize(points)
        elif cfg_idx == 2:
            # Perturbed golden-angle
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            rng = np.random.RandomState(100 + cfg_idx)
            noise = rng.randn(N, 3) * 0.1
            points = points + noise
            points = _normalize(points)
        elif cfg_idx == 3:
            # Fibonacci with different offset
            i = np.arange(N)
            z = 1.0 - 2.0 * (i + 0.3) / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            # More random starts with different seeds
            rng = np.random.RandomState(200 + cfg_idx * 7)
            points = rng.randn(N, 3)
            points = _normalize(points)
        
        configs.append(points)
    
    return configs

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Generate initial configurations
    configs = _generate_initial_configurations(n_configs=12)
    
    best_points = None
    best_radius = np.inf
    
    for cfg_idx, init_points in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10.0:
            break
        
        # Optimize this configuration
        try:
            # First: Lloyd relaxation
            optimized = _lloyd_relaxation(init_points, n_iter=50, tol=1e-8)
            
            # Second: Facet-based local search (the key improvement)
            optimized = _facet_local_search(optimized, max_iter=100, time_budget=5.0)
            
            # Third: Final Lloyd polish
            optimized = _lloyd_relaxation(optimized, n_iter=30, tol=1e-10)
            
            radius = _compute_covering_radius(optimized)
            
            if radius < best_radius:
                best_radius = radius
                best_points = optimized.copy()
        except Exception:
            continue
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 3.0 or best_points is None:
        if best_points is not None:
            return best_points
        else:
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Final polish with remaining time
    try:
        remaining_time = deadline - (time.time() - start_time) - 2.0
        if remaining_time > 5.0:
            final_points = _facet_local_search(best_points, max_iter=100, time_budget=remaining_time * 0.5)
            final_points = _lloyd_relaxation(final_points, n_iter=20, tol=1e-12)
            final_radius = _compute_covering_radius(final_points)
            if final_radius < best_radius:
                best_points = final_points
    except Exception:
        pass
    
    return best_points
# EVOLVE-BLOCK-END
