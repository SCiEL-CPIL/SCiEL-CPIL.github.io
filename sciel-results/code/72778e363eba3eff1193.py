"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _compute_circumcenters(points):
    """
    Compute the circumcenters of all triangular facets of the convex hull.
    Returns an array of circumcenters (on the unit sphere) and their radii.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        centers = []
        radii = []
        
        for f in facets:
            if len(f) != 3:
                continue
            a, b, c = points[f[0]], points[f[1]], points[f[2]]
            
            # Compute circumcenter of triangle abc on the sphere
            # The circumcenter x satisfies |x|^2=1, x.a = x.b = x.c
            # This is equivalent to x being the normalized cross product of (b-a) and (c-a) 
            # adjusted for the sphere, or solving the linear system.
            
            # A robust way: The circumcenter is the intersection of the sphere and the plane defined by a,b,c.
            # Normal to plane: n = cross(b-a, c-a)
            n = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            n = n / norm_n
            
            # The circumcenter lies on the line: x = a + t * n
            # We need |a + t*n|^2 = 1
            # |a|^2 + 2t(a.n) + t^2|n|^2 = 1
            # 1 + 2t(a.n) + t^2 = 1  (since |a|=1, |n|=1)
            # t^2 + 2t(a.n) = 0
            # t(t + 2(a.n)) = 0
            # t = 0 or t = -2(a.n)
            # t=0 gives a, which is not the circumcenter unless the triangle is degenerate.
            # So t = -2 * np.dot(a, n)
            
            t = -2.0 * np.dot(a, n)
            x = a + t * n
            norm_x = np.linalg.norm(x)
            if norm_x < 1e-10:
                continue
            x = x / norm_x
            
            # Check if x is on the correct side (outward normal)
            # The facet normal should point outward.
            # If dot(x, n) < 0, flip x? No, the circumcenter is unique.
            # However, for covering radius, we care about the distance from x to a,b,c.
            # The circumradius is the angular distance from x to a.
            
            # Ensure x is the "far" point from the origin in the direction of the facet normal?
            # Actually, the circumcenter of a spherical triangle is the point on the sphere equidistant from the vertices.
            # There are two such points: x and -x. We want the one that is the center of the empty cap.
            # The empty cap center is the one that is NOT inside the convex hull.
            # Since the convex hull contains the origin (points are on sphere, hull wraps around),
            # the circumcenter of a facet is the point on the sphere in the direction of the facet normal.
            
            # Let's verify: The normal n calculated as cross(b-a, c-a) might be inward or outward.
            # Let's check the dot product with the centroid of the triangle.
            centroid = (a + b + c) / 3.0
            if np.dot(n, centroid) < 0:
                n = -n
                t = -2.0 * np.dot(a, n)
                x = a + t * n
                norm_x = np.linalg.norm(x)
                if norm_x > 1e-10:
                    x = x / norm_x
            
            # Calculate radius
            cos_r = np.clip(np.dot(x, a), -1.0, 1.0)
            r = np.arccos(cos_r)
            
            centers.append(x)
            radii.append(r)
            
        if len(centers) == 0:
            return np.array([]), np.array([])
            
        return np.array(centers), np.array(radii)
    except Exception:
        return np.array([]), np.array([])

def _covering_radius_fast(points):
    """
    Compute the covering radius in degrees.
    """
    centers, radii = _compute_circumcenters(points)
    if len(radii) == 0:
        return np.inf
    return np.degrees(np.max(radii))

def _refine_local(points, max_time=5.0):
    """
    Refine points by moving them towards the circumcenters of the worst facets.
    """
    start_time = time.time()
    current_points = points.copy()
    current_radius = _covering_radius_fast(current_points)
    
    # Iterate: find worst facet, move its vertices towards its circumcenter
    for _ in range(50):
        if time.time() - start_time > max_time:
            break
        
        centers, radii = _compute_circumcenters(current_points)
        if len(radii) == 0:
            break
            
        worst_idx = np.argmax(radii)
        worst_radius = radii[worst_idx]
        worst_center = centers[worst_idx]
        
        # Find the facet corresponding to this center?
        # We need to map centers back to facets.
        # Let's re-compute to get the facet indices for the worst one.
        try:
            hull = ConvexHull(current_points)
            facets = hull.simplices
            # Re-calculate to find which facet has this circumcenter
            # This is inefficient. Let's just move ALL points slightly towards the global worst center? No.
            # Better: Move the vertices of the worst facet.
            
            # We need to identify the facet. Let's just iterate through facets again to find the match.
            target_facet = None
            for f in facets:
                if len(f) != 3:
                    continue
                a, b, c = current_points[f[0]], current_points[f[1]], current_points[f[2]]
                n = np.cross(b - a, c - a)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                n = n / norm_n
                centroid = (a + b + c) / 3.0
                if np.dot(n, centroid) < 0:
                    n = -n
                t = -2.0 * np.dot(a, n)
                x = a + t * n
                norm_x = np.linalg.norm(x)
                if norm_x > 1e-10:
                    x = x / norm_x
                    # Check if this x is close to worst_center
                    if np.dot(x, worst_center) > 0.9999:
                        target_facet = f
                        break
                        
            if target_facet is None:
                break
                
            # Move vertices of target_facet towards worst_center
            improved = False
            for i in target_facet:
                p_i = current_points[i]
                # Direction towards center
                dir_vec = worst_center - p_i
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir < 1e-10:
                    continue
                dir_vec = dir_vec / norm_dir
                
                # Try small steps
                for step in [0.05, 0.02, 0.01, 0.005]:
                    new_p = p_i + step * dir_vec
                    new_p /= np.linalg.norm(new_p)
                    
                    candidate = current_points.copy()
                    candidate[i] = new_p
                    
                    r = _covering_radius_fast(candidate)
                    if r < current_radius:
                        current_points = candidate
                        current_radius = r
                        improved = True
                        break
                        
            if not improved:
                break
                
        except Exception:
            break
            
    return current_points, current_radius

def _spectral_init():
    """
    Generate initial points using a spectral method approximation.
    We maximize the trace of a repulsion matrix or similar.
    For simplicity, we use a random initialization with a repulsion step.
    """
    points = np.random.randn(N, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Simple repulsion to spread them out
    for _ in range(100):
        # Compute pairwise distances
        dots = points @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        # Force: repulsive for close points
        # F_i = sum_j (points_j - points_i) * exp(-d_ij^2 / sigma^2)
        sigma = 0.1
        weights = np.exp(-dists**2 / sigma**2)
        np.fill_diagonal(weights, 0)
        
        # Sum of weighted vectors
        # F_i = sum_j weights[i,j] * (points_j - points_i)
        # = sum_j weights[i,j]*points_j - points_i * sum_j weights[i,j]
        sum_j = weights @ points
        sum_w = weights.sum(axis=1, keepdims=True)
        forces = sum_j - points * sum_w
        
        # Normalize forces to be tangent to sphere
        # Project force onto tangent space: F_tan = F - (F.p)p
        dots_fp = np.einsum('ij,ij->i', forces, points)
        forces_tan = forces - dots_fp[:, np.newaxis] * points
        
        # Update
        step = 0.01
        points += step * forces_tan
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_radius = np.inf
    
    # Multiple starts
    for i in range(5):
        if time.time() > deadline - 20.0:
            break
            
        # Mix of random and spectral init
        if i == 0:
            # Fibonacci spiral
            idx = np.arange(N) + 0.5
            z = 1.0 - 2.0 * idx / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * idx
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            np.random.seed(i * 123 + 456)
            points = _spectral_init()
            
        # Refine
        refined, radius = _refine_local(points, max_time=20.0)
        
        if radius < best_radius:
            best_radius = radius
            best_points = refined
            
    # Final polish
    if best_points is not None and time.time() < deadline - 5.0:
        final_points, final_radius = _refine_local(best_points, max_time=30.0)
        if final_radius < best_radius:
            best_points = final_points
            best_radius = final_radius
            
    return best_points
# EVOLVE-BLOCK-END
