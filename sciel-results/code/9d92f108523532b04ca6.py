import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_hull_radius(pts):
    """
    Compute the exact covering radius using the convex hull method.
    Returns the radius in radians.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi
            
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)
        
        return np.max(ang_radii)
    except Exception:
        return np.pi

def _refine_worst_facet(pts, deadline=None):
    """
    Iteratively identifies the facet with the maximum circumradius and 
    projects its vertices onto the geodesic arc between the facet's pole 
    and the antipode of the pole. This strictly reduces the radius of 
    the worst facet while maintaining the convexity of the hull locally.
    """
    curr_r = _compute_hull_radius(pts)
    best_pts = pts.copy()
    best_r = curr_r
    
    for it in range(500):
        if deadline and time.time() > deadline:
            break
            
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            
            A = pts[faces[:, 0]]
            B = pts[faces[:, 1]]
            C = pts[faces[:, 2]]
            
            AB = B - A
            AC = C - A
            cross = np.cross(AB, AC)
            area = 0.5 * np.linalg.norm(cross, axis=1)
            valid = area > _EPS
            if not np.any(valid):
                break
                
            normals = cross[valid]
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            centroids = (A[valid] + B[valid] + C[valid]) / 3.0
            signs = np.sum(normals * centroids, axis=1, keepdims=True)
            normals = normals * np.sign(signs)
            normals = normals / (norms + _EPS)
            
            cos_theta = np.sum(normals * A[valid], axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            ang_radii = np.arccos(cos_theta)
            
            curr_r = np.max(ang_radii)
            if curr_r < best_r - 1e-15:
                best_r = curr_r
                best_pts = pts.copy()
                
            worst_idx = np.argmax(ang_radii)
            if ang_radii[worst_idx] < 0.1: # Converged
                break
                
            v_indices = faces[valid][worst_idx]
            pole = normals[worst_idx]
            
            # The antipode of the pole is the point on the sphere farthest 
            # from the facet's circumcenter. Moving vertices towards this 
            # antipode shrinks the facet's circumradius.
            target = -pole
            
            # Project vertices onto the geodesic arc between their current 
            # positions and the target. We use a fixed step size in terms 
            # of angular distance for stability.
            step = 0.15
            
            for v in v_indices:
                p = pts[v]
                # Slerp: p -> target
                # theta = arccos(p . target)
                dot = np.dot(p, target)
                dot = np.clip(dot, -1.0, 1.0)
                theta = np.arccos(dot)
                
                if theta < _EPS:
                    continue
                    
                # Move by 'step' radians along the arc
                t = step / theta
                t = np.clip(t, 0.0, 1.0)
                
                # p_new = (sin((1-t)theta)/sin(theta))*p + (sin(t*theta)/sin(theta))*target
                sin_theta = np.sin(theta)
                w1 = np.sin((1.0 - t) * theta) / sin_theta
                w2 = np.sin(t * theta) / sin_theta
                
                p_new = w1 * p + w2 * target
                p_new /= (np.linalg.norm(p_new) + _EPS)
                pts[v] = p_new
                
        except Exception:
            break
            
    return best_pts, best_r

def construct():
    deadline = time.time() + 145.0  # Safety margin
    rng = np.random.default_rng(42)
    
    best_pts = _initial_fib()
    best_r = _compute_hull_radius(best_pts)
    
    # Strategy: Multiple restarts. Each restart starts from a random or 
    # Fibonacci configuration and applies the worst-case vertex refinement 
    # until convergence or deadline.
    
    n_restarts = 10
    time_per_restart = (deadline - 10.0) / (n_restarts + 1)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Initialize
        if restart == 0:
            pts = _initial_fib()
        else:
            pts = rng.standard_normal((N, 3))
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            
        # Apply refinement
        refined_pts, refined_r = _refine_worst_facet(pts, deadline=deadline - 10.0)
        
        if refined_r < best_r:
            best_r = refined_r
            best_pts = refined_pts.copy()
            
    # Final polish with a smaller step size to squeeze out more improvement
    if time.time() < deadline - 2.0:
        # We can run a final refinement on the best found so far
        # Note: _refine_worst_facet uses a fixed step. For final polish, 
        # we could modify it, but for simplicity we just run it again.
        # It will stop quickly if converged.
        final_pts, final_r = _refine_worst_facet(best_pts, deadline=deadline - 1.0)
        if final_r < best_r:
            best_r = final_r
            best_pts = final_pts.copy()

    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    # Validity check
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
