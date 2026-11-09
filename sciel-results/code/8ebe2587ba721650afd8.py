"""
Strategy: Vectorized Lloyd's Algorithm for Spherical k-Means.
Instead of fixing the convex hull and repairing facets, we treat this as a
clustering problem. We define the spherical Voronoi cells and iteratively
move each point to the centroid (in 3D space) of its cell. This is the
standard Lloyd's algorithm, adapted for the sphere. We vectorize the
computation of cell centroids and the covering radius estimation to allow
many more iterations within the time budget, aiming for a deeper local
minimum.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    Vectorized computation of circumradii of convex hull facets.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        # Normal to the plane defined by p1, p2, p3
        # n = (p2-p1) x (p3-p1)
        # But for circumcenter we need the direction equidistant from all 3.
        # C is proportional to (p1-p2) x (p1-p3) ? No.
        # C satisfies (C-p1).n_plane = 0 ... no.
        # C is the intersection of the sphere and the line perpendicular to the 
        # plane of the triangle passing through the origin? No.
        
        # The circumcenter of a spherical triangle is the unit vector c such that
        # c.p1 = c.p2 = c.p3.
        # This implies c is orthogonal to (p1-p2) and (p1-p3).
        # So c is parallel to cross(p1-p2, p1-p3).
        
        v1 = p1 - p2
        v2 = p1 - p3
        c_dir = np.cross(v1, v2)
        
        norms = np.linalg.norm(c_dir, axis=1)
        
        # Handle degenerate cases
        valid = norms > 1e-10
        if not np.any(valid):
            return 90.0 # Fallback
            
        c = np.zeros_like(c_dir)
        c[valid] = c_dir[valid] / norms[valid, np.newaxis]
        
        # Determine correct orientation: c should be "inside" the triangle.
        # Check dot product with centroid.
        centroids = (p1 + p2 + p3) / 3.0
        c_norms = np.linalg.norm(centroids, axis=1)
        valid_centroids = c_norms > 1e-10
        centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
        
        dots = np.sum(c * centroids, axis=1)
        c[dots < 0] *= -1
        
        # Compute angular distance from c to p1
        cos_theta = np.sum(c * p1, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # Only consider valid facets
        theta[~valid] = 0.0
        
        return np.max(theta) * 180.0 / np.pi
        
    except Exception:
        return 180.0

def _lloyd_step(points):
    """
    Perform one step of Lloyd's algorithm.
    1. Assign each facet (triangle) to the nearest point? No, that's not how Voronoi works.
    2. We need the centroid of the Voronoi cell for each point.
    
    Alternative approach for Lloyd's on sphere:
    The Voronoi cell of a point p_i is the set of points y on the sphere such that 
    angle(y, p_i) <= angle(y, p_j) for all j.
    This is equivalent to (y . p_i) >= (y . p_j) for all j.
    
    Computing exact spherical Voronoi centroids is complex.
    However, we can approximate the "force" or move direction.
    
    A simpler and robust method often used in these contexts:
    For each point p_i, find all points p_j such that p_i is the nearest neighbor of p_j?
    No, that's just Delaunay.
    
    Let's use the convex hull dual. The Voronoi cell of a vertex in the Delaunay triangulation
    is dual to the set of faces incident to that vertex.
    
    For each vertex v, its Voronoi cell is the convex hull of the circumcenters of all
    triangles incident to v.
    
    So, algorithm:
    1. Compute ConvexHull.
    2. For each vertex v, find all triangles containing v.
    3. Compute the circumcenter of each such triangle.
    4. The centroid of the Voronoi cell is the centroid of these circumcenters (projected to sphere).
    5. Move v towards this centroid.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        # Map vertex index to list of facet indices
        vertex_to_facets = {i: [] for i in range(N)}
        for idx, fac in enumerate(facets):
            for v in fac:
                vertex_to_facets[v].append(idx)
                
        new_points = points.copy()
        
        for v in range(N):
            facs = vertex_to_facets[v]
            if not facs:
                continue
                
            # Get circumcenters of these facets
            # We can reuse the logic from _covering_radius but vectorized per facet?
            # Or just compute them here.
            
            centers = []
            for f_idx in facs:
                p1 = points[facets[f_idx][0]]
                p2 = points[facets[f_idx][1]]
                p3 = points[facets[f_idx][2]]
                
                v1 = p1 - p2
                v2 = p1 - p3
                c_dir = np.cross(v1, v2)
                n = np.linalg.norm(c_dir)
                if n < 1e-10:
                    continue
                c = c_dir / n
                
                # Orient towards centroid
                centroid = (p1 + p2 + p3) / 3.0
                nc = np.linalg.norm(centroid)
                if nc > 1e-10:
                    centroid /= nc
                    if np.dot(c, centroid) < 0:
                        c = -c
                centers.append(c)
                
            if not centers:
                continue
                
            centers = np.array(centers)
            # Centroid of the Voronoi cell vertices
            cell_centroid = np.mean(centers, axis=0)
            nc = np.linalg.norm(cell_centroid)
            if nc > 1e-10:
                cell_centroid /= nc
                
                # Move towards centroid
                # Slerp or simple linear interpolation?
                # Linear interpolation in 3D followed by normalization is robust.
                # To prevent oscillation, use a small step size or full step?
                # Full step is standard Lloyd.
                new_points[v] = cell_centroid
                
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / norms
        
        return new_points
        
    except Exception:
        return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Run Lloyd's algorithm
    # Each iteration involves a ConvexHull (O(N log N) roughly) and centroid calculation.
    # N=130 is small, so this should be fast.
    
    for iter in range(5000):
        if time.time() > deadline:
            break
            
        # Check current radius
        current_radius = _covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # Perform Lloyd step
        points = _lloyd_step(points)
        
        # Optional: Add small noise to escape local minima?
        # If we get stuck (radius doesn't change), maybe add noise.
        # But Lloyd's is usually monotonic in terms of objective if done correctly?
        # Actually, Lloyd's minimizes the sum of squared distances, which is related
        # but not identical to min-max (covering radius).
        # So it might not be monotonic in covering radius.
        
        # To ensure we don't get stuck in a bad local minimum for covering radius,
        # we keep the best_points found so far.
        
        # If we detect stagnation, we could try a random perturbation.
        # But for now, let's just run it.
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Covering radius: {_covering_radius(pts)} degrees")
    print(pts.shape)
