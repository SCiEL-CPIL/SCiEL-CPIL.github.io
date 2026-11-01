"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import scipy.optimize as opt

def _circumradius_facet(pts):
    """
    Compute the circumradius of a facet defined by 3 points on the unit sphere.
    The circumradius of the spherical triangle is the angle between the normal
    to the plane of the triangle and the center of the sphere (origin).
    Wait, the circumradius of the spherical triangle is the distance from the
    center of the sphere to the circumcenter of the planar triangle?
    No. The covering radius is the angular distance.
    For a facet with vertices v1, v2, v3 on the unit sphere, the circumcenter
    of the spherical triangle is the point on the sphere in the direction of
    the normal vector to the plane defined by v1, v2, v3.
    The angular distance from the origin to this circumcenter is 0? No.
    The circumcenter of the spherical triangle is the point c on the sphere
    such that the angular distance to v1, v2, v3 is equal.
    This point is in the direction of the cross product of (v2-v1) and (v3-v1)?
    Actually, the circumcenter of the spherical triangle is the unit vector
    in the direction of the normal to the plane passing through v1, v2, v3.
    Let n = cross(v2-v1, v3-v1).
    The circumcenter is n / ||n||.
    The angular distance (circumradius) is the angle between n/||n|| and any of the vertices, say v1.
    cos(theta) = dot(n/||n||, v1).
    So theta = arccos(dot(n, v1) / ||n||).
    """
    v1, v2, v3 = pts
    v12 = v2 - v1
    v13 = v3 - v1
    n = np.cross(v12, v13)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-10:
        return 0.0
    # The circumcenter is n / norm_n
    # The distance is the angle between n/norm_n and v1
    cos_theta = np.dot(n, v1) / norm_n
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)

def _covering_radius_exact(pts):
    """
    Compute the exact covering radius in radians.
    The covering radius is the maximum circumradius of the facets of the convex hull.
    """
    try:
        hull = ConvexHull(pts)
        max_rad = 0.0
        for simplex in hull.simplices:
            rad = _circumradius_facet(pts[simplex])
            if rad > max_rad:
                max_rad = rad
        return max_rad
    except Exception:
        # Fallback if hull fails (e.g., coplanar points)
        # Use a grid-based estimate as fallback
        return np.pi / 2.0

def _objective(pts_flat):
    """
    Objective function for optimization.
    We minimize the sum of squared circumradii to make the objective smooth.
    This tends to minimize the max circumradius as well.
    """
    pts = pts_flat.reshape(-1, 3)
    # Normalize to ensure points are on sphere (helps optimizer)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    pts = pts / norms
    
    try:
        hull = ConvexHull(pts)
        radii = []
        for simplex in hull.simplices:
            rad = _circumradius_facet(pts[simplex])
            radii.append(rad)
        radii = np.array(radii)
        # Minimize sum of squares to get a smooth objective
        return np.sum(radii**2)
    except Exception:
        return 100.0

def _initial_fib():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    """Construct 130 points on the unit sphere to minimize the covering radius."""
    start_time = time.time()
    max_time = 120.0
    
    best_pts = None
    best_radius = np.inf
    
    # Strategy: Multiple restarts with Powell optimization
    # We use a few different initial configurations
    
    configs = []
    # 1. Fibonacci spiral
    configs.append(_initial_fib())
    
    # 2. Random points
    np.random.seed(42)
    random_pts = np.random.randn(N, 3)
    random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
    configs.append(random_pts)
    
    # 3. Another random seed
    np.random.seed(123)
    random_pts2 = np.random.randn(N, 3)
    random_pts2 /= np.linalg.norm(random_pts2, axis=1, keepdims=True)
    configs.append(random_pts2)

    for init_pts in configs:
        if time.time() - start_time > max_time - 20.0:
            break
            
        # Flatten for optimizer
        x0 = init_pts.ravel()
        
        # Use Powell method, which is derivative-free and robust
        # maxiter is a safeguard, but we rely on time check
        # We can't easily break out of minimize based on time, 
        # so we limit iterations or use a callback.
        
        def callback(xk):
            if time.time() - start_time > max_time - 10.0:
                raise RuntimeError("Time limit exceeded")
            
        try:
            res = opt.minimize(_objective, x0, method='Powell', 
                               options={'maxiter': 500, 'disp': False},
                               callback=callback)
            opt_pts = res.x.reshape(-1, 3)
            # Normalize
            opt_pts /= np.linalg.norm(opt_pts, axis=1, keepdims=True)
            
            # Evaluate exact radius
            rad = _covering_radius_exact(opt_pts)
            
            if rad < best_radius:
                best_radius = rad
                best_pts = opt_pts.copy()
                
        except RuntimeError:
            # Time limit hit during optimization
            # Try to use the best point from the last callback? 
            # Powell doesn't store intermediate points easily in the result object if interrupted.
            # We'll just skip this restart if it takes too long.
            pass
        except Exception as e:
            pass

    # If no optimization succeeded or improved, use the best initial
    if best_pts is None:
        best_pts = _initial_fib()
        best_radius = _covering_radius_exact(best_pts)
        
    # Final refinement: A few steps of local search using the exact radius
    # Move points towards the center of their Voronoi cells (circumcenters of adjacent facets)
    # This is a heuristic improvement.
    
    for _ in range(5):
        if time.time() - start_time > max_time - 5.0:
            break
        try:
            hull = ConvexHull(best_pts)
            # For each point, find the average of the circumcenters of the facets containing it
            # This is complex. Instead, let's just do a small random perturbation 
            # and accept if it improves the exact radius.
            improved = False
            for i in range(N):
                if time.time() - start_time > max_time - 5.0:
                    break
                # Try small random move
                delta = np.random.randn(3) * 0.01
                new_pt = best_pts[i] + delta
                norm = np.linalg.norm(new_pt)
                if norm < 1e-10:
                    continue
                new_pt /= norm
                new_pts = best_pts.copy()
                new_pts[i] = new_pt
                new_rad = _covering_radius_exact(new_pts)
                if new_rad < best_radius - 1e-10:
                    best_pts = new_pts
                    best_radius = new_rad
                    improved = True
            if not improved:
                break
        except Exception:
            break
            
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-10:
                best_pts[j] += np.random.randn(3) * 0.001
                best_pts[j] /= np.linalg.norm(best_pts[j])
                
    # Final normalization
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
