"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import time
import sys

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate initial configuration using Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure all points are on the unit sphere
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Try multiple random restarts with different initial configurations
    best_points = points
    best_radius = _compute_covering_radius(points)
    
    # If we have time, try a few more restarts
    for restart in range(3):
        if time.time() > deadline - 30:
            break
        
        # Generate a different initial configuration
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        # Add some randomness to the spiral
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i + np.random.randn(N) * 0.5
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        new_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_radius = _compute_covering_radius(new_points)
        if new_radius < best_radius:
            best_points = new_points
            best_radius = new_radius
    
    # Optimize the best configuration found
    if time.time() < deadline - 10:
        best_points = _optimize_configuration(best_points, time.time() + 5, deadline - 5)
    
    return best_points


def _compute_covering_radius(points):
    """
    Compute the covering radius in degrees using the convex hull method.
    The covering radius is the maximum circumradius of all facets of the convex hull.
    """
    try:
        hull = ConvexHull(points)
        max_radius = 0.0
        
        for simplex in hull.simplices:
            pts = points[simplex]
            if len(pts) < 3:
                continue
            
            # For a simplex, the circumcenter can be found by solving
            # for the point equidistant from all vertices
            if len(pts) == 3:
                # Triangle: use the formula for circumcenter
                a, b, c = pts
                # Find circumcenter using cross products
                v1 = b - a
                v2 = c - a
                n = np.cross(v1, v2)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-15:
                    continue
                n = n / n_norm
                
                # Circumcenter is along the normal from the plane
                # Find the point on the line through the normal that is equidistant
                # from all three points
                # |x - a|^2 = |x - b|^2 = |x - c|^2
                # Let x = a + t*n, solve for t
                # |a + t*n - a|^2 = |a + t*n - b|^2
                # t^2 = |a - b + t*n|^2 = |a-b|^2 + 2t*n·(a-b) + t^2
                # 0 = |a-b|^2 + 2t*n·(a-b)
                # t = -|a-b|^2 / (2*n·(a-b))
                ab = a - b
                ab_sq = np.dot(ab, ab)
                ndotab = np.dot(n, ab)
                if abs(ndotab) < 1e-15:
                    continue
                t = -ab_sq / (2.0 * ndotab)
                circumcenter = a + t * n
                circumcenter_norm = np.linalg.norm(circumcenter)
                if circumcenter_norm < 1e-15:
                    continue
                circumcenter = circumcenter / circumcenter_norm
                
                # The covering radius is the angular distance from the circumcenter
                # to any of the vertices
                cos_r = np.clip(np.dot(circumcenter, a), -1.0, 1.0)
                radius = np.arccos(cos_r)
                max_radius = max(max_radius, radius)
            
            elif len(pts) == 4:
                # Tetrahedron: find circumcenter
                a, b, c, d = pts
                # Solve for x such that |x-a|^2 = |x-b|^2 = |x-c|^2 = |x-d|^2
                # This gives linear equations
                # |x|^2 - 2x·a + |a|^2 = |x|^2 - 2x·b + |b|^2
                # 2x·(b-a) = |b|^2 - |a|^2
                # Since all points are on unit sphere, |a|^2 = |b|^2 = 1
                # So 2x·(b-a) = 0, etc.
                # x·(b-a) = 0, x·(c-a) = 0, x·(d-a) = 0
                # This means x is perpendicular to all edges from a
                # x is along the normal to the plane through b,c,d from a
                # Actually, for points on the unit sphere, the circumcenter of a tetrahedron
                # is the point x such that x·(b-a) = x·(c-a) = x·(d-a) = 0
                # Wait, that's not right. Let me reconsider.
                
                # For points on the unit sphere, |a|=|b|=|c|=|d|=1
                # |x-a|^2 = |x-b|^2 => -2x·a + 1 = -2x·b + 1 => x·(b-a) = 0
                # So x is perpendicular to (b-a), (c-a), (d-a)
                # These three vectors span R^3 (for a non-degenerate tetrahedron)
                # So x = 0, which is not on the sphere.
                
                # Actually, I think I'm confusing myself. Let me use a different approach.
                # The covering radius for a facet is the angular distance from the 
                # circumcenter of the facet to the farthest vertex, but only if the 
                # circumcenter is inside the cap.
                
                # For a tetrahedron on the unit sphere, the circumcenter of the facet
                # (which is a triangle) is what we need. But the facet has 4 vertices.
                # Actually, the convex hull facets in 3D are triangles, so we should 
                # only have triangular facets. Let me check if ConvexHull gives 
                # only triangular facets in 3D.
                
                # In 3D, ConvexHull.simplices should only contain triangles.
                # So this case might not occur. But let's handle it just in case.
                # For a set of 4 points on the sphere, the worst point is the 
                # circumcenter of the triangle formed by 3 of them, if it's inside
                # the cap.
                
                # Actually, for the covering radius, we need to find the point on 
                # the sphere that is farthest from all the facet vertices. For a 
                # triangular facet, this is the circumcenter (if it's inside the cap).
                # For a quadrilateral facet, it's more complex.
                
                # Let's just use the centroid for non-triangular facets as an approximation
                centroid = np.mean(pts, axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm < 1e-15:
                    continue
                centroid = centroid / centroid_norm
                
                max_dist = 0.0
                for p in pts:
                    cos_r = np.clip(np.dot(centroid, p), -1.0, 1.0)
                    dist = np.arccos(cos_r)
                    max_dist = max(max_dist, dist)
                max_radius = max(max_radius, max_dist)
            
            else:
                # For higher-dimensional simplices (shouldn't occur in 3D)
                centroid = np.mean(pts, axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm < 1e-15:
                    continue
                centroid = centroid / centroid_norm
                
                max_dist = 0.0
                for p in pts:
                    cos_r = np.clip(np.dot(centroid, p), -1.0, 1.0)
                    dist = np.arccos(cos_r)
                    max_dist = max(max_dist, dist)
                max_radius = max(max_radius, max_dist)
        
        return np.degrees(max_radius)
    except Exception:
        return np.inf


def _optimize_configuration(points, start_time, deadline):
    """
    Optimize the configuration using a gradient-free method.
    We'll use a simple approach: try to move each point slightly in the direction
    that reduces the covering radius.
    """
    best_points = points.copy()
    best_radius = _compute_covering_radius(best_points)
    
    # Use a simple coordinate descent with random perturbations
    step_size = 0.1
    
    while time.time() < deadline:
        improved = False
        
        for i in range(N):
            if time.time() > deadline:
                break
            
            # Try moving point i in a random direction
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            
            # Try a few step sizes
            for step in [step_size, step_size * 0.5, step_size * 0.1]:
                new_point = best_points[i] + step * direction
                new_point_norm = np.linalg.norm(new_point)
                if new_point_norm < 1e-15:
                    continue
                new_point = new_point / new_point_norm
                
                candidate = best_points.copy()
                candidate[i] = new_point
                
                radius = _compute_covering_radius(candidate)
                if radius < best_radius:
                    best_points = candidate
                    best_radius = radius
                    improved = True
                    break
        
        if not improved:
            # If no improvement, reduce step size
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return best_points


if __name__ == "__main__":
    result = construct()
    print(f"Shape: {result.shape}")
    print(f"Covering radius: {_compute_covering_radius(result)} degrees")
