"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _covering_radius(points):
    """Compute the covering radius in radians by checking all hull facets."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # fallback: worst case
    
    max_rad = 0.0
    for simplex in hull.simplices:
        # Get the facet vertices
        facet_pts = points[simplex]
        # Compute the circumcenter of the facet
        # For a triangle (2-simplex), the circumcenter is equidistant from all 3 vertices
        # We project the circumcenter onto the unit sphere
        # The circumcenter of a spherical triangle is the point on the sphere equidistant from all 3 vertices
        
        # Method: find the point on the sphere that is equidistant from all facet vertices
        # This is the intersection of the sphere with the plane perpendicular to each edge at its midpoint
        # Simpler: use the fact that for points on a unit sphere, the circumcenter of the spherical triangle
        # is proportional to the sum of the vertices (for small triangles) but more accurately:
        
        # The circumcenter c satisfies |c - v_i|^2 = |c - v_j|^2 for all i,j
        # Since |c|=1 and |v_i|=1, this means c·v_i = c·v_j for all i,j
        # So c is in the null space of (v_i - v_j) for all pairs
        
        # For 3 points, we need c·v_0 = c·v_1 = c·v_2
        # c·(v_0 - v_1) = 0 and c·(v_0 - v_2) = 0
        # So c is perpendicular to (v_0-v_1) and (v_0-v_2)
        # c is parallel to cross(v_0-v_1, v_0-v_2)
        
        d1 = facet_pts[0] - facet_pts[1]
        d2 = facet_pts[0] - facet_pts[2]
        c = np.cross(d1, d2)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-12:
            # Degenerate case
            continue
        c /= c_norm
        
        # Make sure c is on the "outside" (same side as the facet normal pointing outward)
        # The facet normal should point away from the origin
        # Check: c should have positive dot product with the centroid of the facet
        centroid = np.mean(facet_pts, axis=0)
        if np.dot(c, centroid) < 0:
            c = -c
        
        # The angular distance from c to any facet vertex
        # This is the circumradius of the facet
        dot_val = np.dot(c, facet_pts[0])
        dot_val = np.clip(dot_val, -1.0, 1.0)
        rad = np.arccos(dot_val)
        
        if rad > max_rad:
            max_rad = rad
    
    return max_rad


def _lloyd_step(points):
    """Perform one Lloyd iteration: move each point to the circumcenter of its largest facet."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return points.copy()
    
    new_points = points.copy()
    
    # For each vertex, find the largest facet adjacent to it
    # and move the point toward the circumcenter of that facet
    for i in range(N):
        # Find facets containing vertex i
        facet_indices = []
        for f_idx, simplex in enumerate(hull.simplices):
            if i in simplex:
                facet_indices.append(f_idx)
        
        if not facet_indices:
            continue
        
        # Find the facet with the largest circumradius
        best_rad = -1
        best_c = None
        for f_idx in facet_indices:
            simplex = hull.simplices[f_idx]
            facet_pts = points[simplex]
            
            d1 = facet_pts[0] - facet_pts[1]
            d2 = facet_pts[0] - facet_pts[2]
            c = np.cross(d1, d2)
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-12:
                continue
            c /= c_norm
            
            centroid = np.mean(facet_pts, axis=0)
            if np.dot(c, centroid) < 0:
                c = -c
            
            dot_val = np.dot(c, facet_pts[0])
            dot_val = np.clip(dot_val, -1.0, 1.0)
            rad = np.arccos(dot_val)
            
            if rad > best_rad:
                best_rad = rad
                best_c = c
        
        if best_c is not None:
            # Move point i toward best_c with a small step
            # Use a step size that ensures we don't overshoot
            step = 0.1
            new_points[i] = points[i] + step * best_c
            # Normalize
            norm = np.linalg.norm(new_points[i])
            if norm > 1e-12:
                new_points[i] /= norm
    
    return new_points


def _refine(points, time_budget):
    """Refine the configuration using Lloyd iterations with time budget."""
    start_time = time.time()
    best_points = points.copy()
    best_rad = _covering_radius(points)
    
    # Adaptive step size
    step = 0.1
    
    for iteration in range(500):
        if time.time() - start_time > time_budget:
            break
        
        new_points = _lloyd_step(best_points)
        
        # Check for improvement
        new_rad = _covering_radius(new_points)
        
        if new_rad < best_rad:
            best_points = new_points
            best_rad = new_rad
            # Reduce step size slowly
            step = max(0.01, step * 0.95)
        else:
            # Try smaller step
            step = max(0.005, step * 0.8)
            if step <= 0.005 and new_rad >= best_rad:
                # Try even smaller steps a few more times
                for small_step in [0.002, 0.001, 0.0005]:
                    new_points = best_points.copy()
                    try:
                        hull = ConvexHull(best_points)
                        for i in range(N):
                            facet_indices = []
                            for f_idx, simplex in enumerate(hull.simplices):
                                if i in simplex:
                                    facet_indices.append(f_idx)
                            if not facet_indices:
                                continue
                            best_f_rad = -1
                            best_c = None
                            for f_idx in facet_indices:
                                simplex = hull.simplices[f_idx]
                                facet_pts = best_points[simplex]
                                d1 = facet_pts[0] - facet_pts[1]
                                d2 = facet_pts[0] - facet_pts[2]
                                c = np.cross(d1, d2)
                                c_norm = np.linalg.norm(c)
                                if c_norm < 1e-12:
                                    continue
                                c /= c_norm
                                centroid = np.mean(facet_pts, axis=0)
                                if np.dot(c, centroid) < 0:
                                    c = -c
                                dot_val = np.dot(c, facet_pts[0])
                                dot_val = np.clip(dot_val, -1.0, 1.0)
                                rad = np.arccos(dot_val)
                                if rad > best_f_rad:
                                    best_f_rad = rad
                                    best_c = c
                            if best_c is not None:
                                new_points[i] = best_points[i] + small_step * best_c
                                norm = np.linalg.norm(new_points[i])
                                if norm > 1e-12:
                                    new_points[i] /= norm
                    except Exception:
                        break
                    
                    small_rad = _covering_radius(new_points)
                    if small_rad < best_rad:
                        best_points = new_points
                        best_rad = small_rad
                        break
        
        # Check for stagnation
        if best_rad < 0.1975:  # ~11.31 degrees
            break
    
    return best_points, best_rad


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin before 150s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Refine using Lloyd iterations
    points, _ = _refine(points, deadline - (time.time() - start_time))
    
    # Final check: ensure all points are on the unit sphere
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    return points
# EVOLVE-BLOCK-END
