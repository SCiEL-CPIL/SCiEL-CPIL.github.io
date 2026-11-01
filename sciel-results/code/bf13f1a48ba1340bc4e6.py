import numpy as np
import time
from scipy.spatial import Delaunay, ConvexHull

N = 130

def _compute_circumradii(points):
    # points: (N, 3)
    # Returns max circumradius
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_r = 0
        for f in facets:
            a, b, c = points[f[0]], points[f[1]], points[f[2]]
            # Circumradius of triangle abc
            # Using formula R = abc / (4 * Area)
            # Side lengths
            a_len = np.linalg.norm(b - c)
            b_len = np.linalg.norm(c - a)
            c_len = np.linalg.norm(a - b)
            # Area
            cross = np.cross(b - a, c - a)
            area = 0.5 * np.linalg.norm(cross)
            if area < 1e-10:
                continue
            r = (a_len * b_len * c_len) / (4 * area)
            if r > max_r:
                max_r = r
        return max_r
    except:
        return np.inf

def _lloyd_iteration(points):
    # points: (N, 3)
    # Returns new points
    try:
        tri = Delaunay(points)
        # For each point, find adjacent triangles
        # tri.simplices contains indices
        # We need to map point index to triangles containing it
        # This is slow if done naively. 
        # Alternative: use the fact that Voronoi vertices are circumcenters.
        # For each point i, sum the circumcenters of triangles containing i.
        
        new_points = np.zeros_like(points)
        counts = np.zeros(N)
        
        # Iterate over all triangles
        for simplex in tri.simplices:
            a, b, c = simplex
            pa, pb, pc = points[a], points[b], points[c]
            # Compute circumcenter
            # Circumcenter is the intersection of perpendicular bisectors.
            # In 3D, it's the point equidistant from a, b, c.
            # It lies in the plane of the triangle.
            # Formula:
            # d = 2 * (a * (b x c) + b * (c x a) + c * (a x b))
            # c = (|a|^2 (b x c) + |b|^2 (c x a) + |c|^2 (a x b)) / d
            # Note: This formula works for the circumcenter in the plane.
            
            a_sq = np.dot(pa, pa)
            b_sq = np.dot(pb, pb)
            c_sq = np.dot(pc, pc)
            
            bc = np.cross(pb, pc)
            ca = np.cross(pc, pa)
            ab = np.cross(pa, pb)
            
            d = 2.0 * (np.dot(pa, bc) + np.dot(pb, ca) + np.dot(pc, ab))
            
            if abs(d) < 1e-10:
                # Degenerate triangle
                continue
                
            circumcenter = (a_sq * bc + b_sq * ca + c_sq * ab) / d
            
            # Add to accumulators
            new_points[a] += circumcenter
            new_points[b] += circumcenter
            new_points[c] += circumcenter
            counts[a] += 1
            counts[b] += 1
            counts[c] += 1
        
        # Average
        mask = counts > 0
        new_points[mask] /= counts[mask, np.newaxis]
        
        # Renormalize
        norms = np.linalg.norm(new_points, axis=1)
        new_points = new_points / norms[:, np.newaxis]
        
        return new_points
    except:
        return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _compute_circumradii(points)
    
    # Lloyd Relaxation
    while time.time() < deadline:
        new_points = _lloyd_iteration(points)
        
        # Check if points are valid (no coincident points, etc.)
        # Simple check: min distance
        # If min dist is too small, break or revert
        dists = np.linalg.norm(new_points[:, np.newaxis, :] - new_points[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < 1e-5:
            # Points collapsed, break
            break
            
        points = new_points
        current_radius = _compute_circumradii(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # Optional: Add small random perturbation to escape local minima
        # But Lloyd is quite stable.
        
    return best_points
