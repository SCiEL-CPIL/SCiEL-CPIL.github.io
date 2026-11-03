"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial
import math

def _compute_covering_radius(points):
    """
    Computes the covering radius in radians.
    Returns the max circumradius of the Delaunay triangles.
    """
    if len(points) < 3:
        return 0.0
    try:
        # Delaunay triangulation of points on the sphere (convex hull facets)
        # scipy.spatial.Delaunay works on 3D points
        tri = scipy.spatial.Delaunay(points)
        simplices = tri.simplices
        
        max_rad = 0.0
        # Precompute for speed? 
        for idx in simplices:
            a = points[idx[0]]
            b = points[idx[1]]
            c = points[idx[2]]
            
            # Normal vector to the plane of the triangle
            # n = (b-a) x (c-a)
            n = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n)
            
            if norm_n < 1e-10:
                # Degenerate triangle, points are collinear or coincident
                # In this case, the "circumcenter" is ill-defined.
                # The covering radius contribution is likely large if points are far apart,
                # but if they are coincident, it's a local issue.
                # For robustness, we treat this as a large radius to penalize it?
                # Or just skip? Skipping might allow coincident points.
                # Let's calculate the max distance to the 3 points from the centroid?
                # Actually, if norm_n is small, the points are close to a great circle.
                # The circumradius is 90 degrees? No.
                # If points are on a great circle, the circumcenter is on the equator?
                # Let's just use a large value to force movement.
                max_rad = max(max_rad, math.pi/2) 
                continue
                
            c_center = n / norm_n
            
            # Ensure c_center is on the "correct" side (facing the triangle)
            # The dot product with the centroid should be positive?
            # Centroid g = (a+b+c)/3
            # If c_center . g < 0, flip c_center
            # Actually, for a triangle on the sphere, the circumcenter is unique.
            # The vector n points to one of the two poles.
            # We want the pole that is closer to the triangle?
            # The distance from c_center to a is arccos(c_center . a).
            # We want the min of the two possible poles?
            # The two poles are n/|n| and -n/|n|.
            # Let's check both.
            cos_r1 = np.dot(c_center, a)
            cos_r2 = np.dot(-c_center, a)
            
            # The circumradius is the angle.
            # We want the angle to be <= 90 deg? No, covering radius can be > 90.
            # But for a valid Delaunay triangle on the sphere, the circumcenter should be inside the triangle?
            # If the triangle is larger than a hemisphere, the circumcenter is on the other side.
            # However, for N=50, triangles are small.
            
            # Let's pick the pole that gives the smaller distance to the vertices?
            # No, the circumradius is defined by the distance to the vertices.
            # Both poles give the same distance to the vertices?
            # |n . a| = |n . b| = |n . c|?
            # Yes, because n is orthogonal to b-a and c-a.
            # n . (b-a) = 0 => n . b = n . a
            # n . (c-a) = 0 => n . c = n . a
            # So n . a = n . b = n . c.
            # So the angle is arccos(n . a / |n|).
            
            # We need to ensure we are measuring the angle correctly.
            # If n . a is negative, the angle is > 90.
            # Is it possible for the covering radius to be > 90?
            # With 50 points, no. The average area is 4*pi/50.
            # Typical radius is ~18 deg.
            
            rad = np.arccos(np.clip(c_center[0]*a[0] + c_center[1]*a[1] + c_center[2]*a[2], -1.0, 1.0))
            
            # Wait, if the triangle is "large", the circumcenter might be the other pole.
            # But for small triangles, n . a is positive.
            # Let's just take the rad.
            
            if rad > max_rad:
                max_rad = rad
                
    except Exception:
        # If triangulation fails, return a large value
        return math.pi
        
    return max_rad

def construct():
    # Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimization
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    # Parameters
    learning_rate = 0.05
    top_k = 5 # Consider top 5 worst triangles for gradient
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Simple local search
    while time.time() - start_time < timeout:
        # Compute Delaunay
        try:
            tri = scipy.spatial.Delaunay(points)
            simplices = tri.simplices
        except Exception:
            # If triangulation fails, break or restart
            break
            
        # Calculate radii and circumcenters for all triangles
        radii = []
        centers = []
        for idx in simplices:
            a = points[idx[0]]
            b = points[idx[1]]
            c = points[idx[2]]
            
            n = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n)
            
            if norm_n < 1e-10:
                radii.append(math.pi)
                centers.append(np.zeros(3))
                continue
                
            c_center = n / norm_n
            # Check orientation
            # We want the circumcenter to be the one that is "inside" the spherical triangle?
            # For small triangles, c_center . a > 0.
            # If c_center . a < 0, the triangle is large (> hemisphere) or we picked the wrong pole.
            # But for N=50, triangles are small.
            # However, if the triangle is obtuse, the circumcenter is outside?
            # On the sphere, the circumcenter is always well-defined as the point equidistant.
            
            cos_r = np.dot(c_center, a)
            # Clip to [-1, 1]
            cos_r = np.clip(cos_r, -1.0, 1.0)
            rad = np.arccos(cos_r)
            
            radii.append(rad)
            centers.append(c_center)
            
        radii = np.array(radii)
        centers = np.array(centers)
        
        # Update best
        current_max_rad = np.max(radii)
        if current_max_rad < best_radius:
            best_radius = current_max_rad
            best_points = points.copy()
            
        # Find indices of top_k largest radii
        # argsort gives indices that would sort the array. 
        # We want the largest, so we take the last k.
        sorted_indices = np.argsort(radii)
        worst_indices = sorted_indices[-top_k:]
        
        # Calculate gradient
        # For each worst triangle, move its vertices towards its circumcenter
        grad = np.zeros_like(points)
        
        for idx_tri in worst_indices:
            idx = simplices[idx_tri]
            c_center = centers[idx_tri]
            rad = radii[idx_tri]
            
            # Weight by radius? Larger radius gets more weight?
            # Or equal weight?
            weight = 1.0 
            
            for i in range(3):
                p_idx = idx[i]
                p = points[p_idx]
                
                # Vector from p to c_center
                v = c_center - p
                # Project onto tangent space at p (remove component along p)
                # proj = v - (v . p) p
                v_dot_p = np.dot(v, p)
                v_tangent = v - v_dot_p * p
                
                grad[p_idx] += weight * v_tangent
                
        # Normalize gradient?
        # Apply update
        # Small random jitter to escape local minima
        if np.random.rand() < 0.1:
            grad += np.random.normal(0, 0.01, (N, 3))
            
        points = points + learning_rate * grad
        
        # Normalize points
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms < 1e-10] = 1.0
        points = points / norms
        
        # Check for coincident points (invalid)
        # If any two points are too close, push them apart
        # This is expensive O(N^2), but N=50 is small
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(points[i] - points[j])
                if dist < 1e-4:
                    # Push apart
                    dir_vec = points[i] - points[j]
                    if np.linalg.norm(dir_vec) < 1e-8:
                        dir_vec = np.random.normal(0, 1, 3)
                    dir_vec = dir_vec / np.linalg.norm(dir_vec)
                    points[i] += 0.1 * dir_vec
                    points[j] -= 0.1 * dir_vec
                    # Re-normalize
                    points[i] /= np.linalg.norm(points[i])
                    points[j] /= np.linalg.norm(points[j])
                    
    return best_points
# EVOLVE-BLOCK-END
