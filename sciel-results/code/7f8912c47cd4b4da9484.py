"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import itertools

def _normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm

def _fibonacci_spiral(n):
    """Generate n points on the unit sphere using a Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _spherical_delaunay_circumcenters(points):
    """
    Compute circumcenters and circumradii of the Delaunay triangles on the sphere.
    Uses ConvexHull on 3D points. Facets of the hull are the Delaunay triangles.
    Returns:
        circumcenters: array of shape (num_facets, 3), normalized to unit sphere.
        circumradii: array of shape (num_facets,), angular distances.
        facet_indices: list of index tuples for each facet.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback if hull fails (e.g., points coplanar or degenerate)
        # Return dummy values
        return np.zeros((0, 3)), np.zeros(0), []
    
    facets = hull.simplices
    num_facets = len(facets)
    circumcenters = np.zeros((num_facets, 3))
    circumradii = np.zeros(num_facets)
    
    for i, facet in enumerate(facets):
        p1 = points[facet[0]]
        p2 = points[facet[1]]
        p3 = points[facet[2]]
        
        # Compute circumcenter of the triangle in 3D
        # Solve for c such that |c - p1|^2 = |c - p2|^2 = |c - p3|^2
        # This implies c is in the plane of the triangle and equidistant.
        # A simpler way: The circumcenter of a triangle with vertices on the unit sphere
        # lies on the line passing through the origin and the vector sum of the vertices?
        # No, that's the centroid.
        # The circumcenter of the triangle formed by p1, p2, p3 on the sphere
        # is the intersection of the sphere and the plane perpendicular to the normal of the triangle?
        # Actually, the circumcenter of the spherical triangle is the point on the sphere
        # equidistant from p1, p2, p3.
        # Let c be the circumcenter. c . p1 = c . p2 = c . p3 = cos(R).
        # c is in the span of p1, p2, p3.
        # We can find the vector v = p1 + p2 + p3. This is not generally the circumcenter.
        # Let's solve the linear system.
        # c = a*p1 + b*p2 + c_coef*p3
        # c . p1 = cos(R) => a + b*(p2.p1) + c_coef*(p3.p1) = cos(R)
        # c . p2 = cos(R) => a*(p1.p2) + b + c_coef*(p3.p2) = cos(R)
        # c . p3 = cos(R) => a*(p1.p3) + b*(p2.p3) + c_coef = cos(R)
        
        p1p2 = np.dot(p1, p2)
        p1p3 = np.dot(p1, p3)
        p2p3 = np.dot(p2, p3)
        
        A = np.array([
            [1.0, p1p2, p1p3],
            [p1p2, 1.0, p2p3],
            [p1p3, p2p3, 1.0]
        ])
        
        # We need to find c such that it is equidistant.
        # Let d = cos(R).
        # The system is:
        # [1, p1p2, p1p3] [a, b, c_coef]^T = d
        # [p1p2, 1, p2p3] [a, b, c_coef]^T = d
        # [p1p3, p2p3, 1] [a, b, c_coef]^T = d
        # This system depends on d. We can eliminate d by subtracting equations.
        # Eq1 - Eq2: a(1-p1p2) + b(p1p2-1) + c_coef(p1p3-p2p3) = 0
        # Eq1 - Eq3: a(1-p1p3) + b(p1p2-p2p3) + c_coef(p1p3-1) = 0
        
        B = np.array([
            [1.0 - p1p2, p1p2 - 1.0, p1p3 - p2p3],
            [1.0 - p1p3, p1p2 - p2p3, p1p3 - 1.0]
        ])
        # We have 2 equations and 3 unknowns (a, b, c_coef).
        # We can set c_coef = 1 and solve for a and b?
        # Or find the null space?
        # Actually, the circumcenter vector direction is given by:
        # v = (|p2|^2 * (p3 x p1) + |p3|^2 * (p1 x p2) + |p1|^2 * (p2 x p3))
        # Since points are on unit sphere, |p|^2 = 1.
        # v = (p3 x p1) + (p1 x p2) + (p2 x p3)
        
        v = np.cross(p3, p1) + np.cross(p1, p2) + np.cross(p2, p3)
        
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            # Degenerate triangle
            circumcenters[i] = _normalize(p1 + p2 + p3)
            circumradii[i] = 0.0
        else:
            c_3d = v / norm_v
            circumcenters[i] = c_3d
            
            # Calculate circumradius (angular distance from c_3d to p1)
            dot_val = np.dot(c_3d, p1)
            # Clip to avoid numerical errors
            dot_val = np.clip(dot_val, -1.0, 1.0)
            circumradii[i] = np.arccos(dot_val)
            
    return circumcenters, circumradii, facets

def construct():
    start_time = time.time()
    deadline = 100.0 # Leave 20s margin for safety
    
    # 1. Initialize
    points = _fibonacci_spiral(N)
    
    best_points = points.copy()
    best_radius = np.inf
    
    # Pre-compute for efficiency? No, points change.
    
    # Optimization Loop
    step_size = 0.05 # Initial step size
    min_step_size = 0.001
    cooling_factor = 0.95
    
    for iteration in range(1000):
        if time.time() > deadline:
            break
            
        circumcenters, circumradii, facets = _spherical_delaunay_circumcenters(points)
        
        if len(circumradii) == 0:
            break
            
        current_max_radius = np.max(circumradii)
        
        if current_max_radius < best_radius:
            best_radius = current_max_radius
            best_points = points.copy()
            
        # Check for convergence
        if current_max_radius < 0.319: # ~18.3 degrees in radians
             # We are very close to the known optimum.
             # We can try to polish further with smaller steps.
             pass
             
        # Determine update for each point
        # For each point i, find the max circumradius among facets incident to i.
        # Map facet index to point indices.
        # facet_indices is a list of tuples (i, j, k)
        
        # Create an array to hold the target circumcenter for each point
        # If a point is in multiple large facets, we can average or pick the worst.
        # Picking the worst is more aggressive.
        
        target_centers = np.zeros_like(points)
        weights = np.zeros(N)
        
        for i, facet in enumerate(facets):
            r = circumradii[i]
            c = circumcenters[i]
            p_idx = facet
            
            # For each point in this facet, add the circumcenter to its target
            # Weight by the radius? Or just count?
            # If we just move towards the circumcenter of the worst facet, it might oscillate.
            # Let's accumulate the circumcenters weighted by how "bad" they are.
            
            # Only consider facets that are significantly large?
            # Or just all of them.
            
            for p in p_idx:
                target_centers[p] += c
                weights[p] += 1.0
                
        # Normalize the target direction for each point
        for i in range(N):
            if weights[i] > 0:
                target_dir = target_centers[i] / weights[i]
                norm_t = np.linalg.norm(target_dir)
                if norm_t > 1e-12:
                    target_dir = target_dir / norm_t
                    # Move point i towards target_dir
                    # Use a small step
                    points[i] = points[i] + step_size * (target_dir - points[i])
                    points[i] = _normalize(points[i])
                else:
                    # If target is zero (unlikely), keep point
                    pass
            
        # Reduce step size
        step_size *= cooling_factor
        if step_size < min_step_size:
             step_size = min_step_size
             
    # Final Polish
    # Run a few more iterations with very small steps to settle
    step_size = 0.001
    for _ in range(50):
        if time.time() > deadline:
            break
        circumcenters, circumradii, facets = _spherical_delaunay_circumcenters(points)
        if len(circumradii) == 0: break
        current_max_radius = np.max(circumradii)
        if current_max_radius < best_radius:
            best_radius = current_max_radius
            best_points = points.copy()
            
        target_centers = np.zeros_like(points)
        weights = np.zeros(N)
        for i, facet in enumerate(facets):
            c = circumcenters[i]
            for p in facet:
                target_centers[p] += c
                weights[p] += 1.0
        for i in range(N):
            if weights[i] > 0:
                target_dir = target_centers[i] / weights[i]
                norm_t = np.linalg.norm(target_dir)
                if norm_t > 1e-12:
                    target_dir = target_dir / norm_t
                    points[i] = points[i] + step_size * (target_dir - points[i])
                    points[i] = _normalize(points[i])
                    
    # Ensure best_points is returned
    return best_points

# EVOLVE-BLOCK-END
