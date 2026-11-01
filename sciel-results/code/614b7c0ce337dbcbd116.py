"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
import scipy.optimize as opt

def _circumradius_facet(v1, v2, v3):
    """Compute the circumradius (angular distance) of a spherical triangle facet."""
    v12 = v2 - v1
    v13 = v3 - v1
    n = np.cross(v12, v13)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-10:
        return 0.0
    cos_theta = np.dot(n, v1) / norm_n
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)

def _all_facet_radii(pts):
    """Compute circumradius for all facets of the convex hull."""
    try:
        hull = ConvexHull(pts)
        radii = np.zeros(len(hull.simplices))
        for i, simplex in enumerate(hull.simplices):
            radii[i] = _circumradius_facet(pts[simplex[0]], pts[simplex[1]], pts[simplex[2]])
        return radii, hull.simplices
    except Exception:
        return np.array([np.pi/2.0]), np.array([[0,1,2]])

def _covering_radius(pts):
    """Compute the exact covering radius in radians."""
    radii, _ = _all_facet_radii(pts)
    return np.max(radii)

def _lloyd_iteration(pts, n_iter=3):
    """Lloyd's algorithm: move each point to the centroid of its Voronoi cell."""
    pts = pts.copy()
    for _ in range(n_iter):
        try:
            # Use Delaunay triangulation on the sphere via 3D
            delaunay = Delaunay(pts)
            # For each point, find its Voronoi cell centroid
            new_pts = pts.copy()
            for i in range(N):
                # Find facets containing point i
                simplices_with_i = [s for s in delaunay.simplices if i in s]
                if len(simplices_with_i) == 0:
                    continue
                # Compute centroid of the Voronoi cell as average of facet circumcenters
                centroids = []
                for s in simplices_with_i:
                    v1, v2, v3 = pts[s[0]], pts[s[1]], pts[s[2]]
                    v12 = v2 - v1
                    v13 = v3 - v1
                    n = np.cross(v12, v13)
                    norm_n = np.linalg.norm(n)
                    if norm_n > 1e-10:
                        centroid = n / norm_n
                        centroids.append(centroid)
                if centroids:
                    avg_centroid = np.mean(centroids, axis=0)
                    norm_avg = np.linalg.norm(avg_centroid)
                    if norm_avg > 1e-10:
                        new_pts[i] = avg_centroid / norm_avg
            pts = new_pts
        except Exception:
            break
    return pts

def _facet_targeted_optimize(pts, max_time_remaining):
    """Optimize by targeting the worst facets specifically."""
    pts = pts.copy()
    start = time.time()
    
    for iteration in range(50):
        if time.time() - start > max_time_remaining - 5:
            break
        
        radii, simplices = _all_facet_radii(pts)
        max_rad = np.max(radii)
        max_idx = np.argmax(radii)
        worst_simplex = simplices[max_idx]
        
        # Get the three points of the worst facet
        i1, i2, i3 = worst_simplex[0], worst_simplex[1], worst_simplex[2]
        
        # Compute the circumcenter of this facet
        v1, v2, v3 = pts[i1], pts[i2], pts[i3]
        v12 = v2 - v1
        v13 = v3 - v1
        n = np.cross(v12, v13)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        circumcenter = n / norm_n
        
        # Move each of the three points slightly toward the circumcenter
        # This should reduce the circumradius
        alpha = 0.1  # Step size
        for idx in [i1, i2, i3]:
            direction = circumcenter - pts[idx]
            direction /= np.linalg.norm(direction) + 1e-10
            new_pt = pts[idx] + alpha * direction
            new_pt /= np.linalg.norm(new_pt)
            pts[idx] = new_pt
        
        # Verify improvement
        new_rad = _covering_radius(pts)
        if new_rad > max_rad + 1e-8:
            # Revert if it got worse
            pts[i1] = v1
            pts[i2] = v2
            pts[i3] = v3
            # Try smaller step
            alpha = 0.05
            for idx in [i1, i2, i3]:
                direction = circumcenter - pts[idx]
                direction /= np.linalg.norm(direction) + 1e-10
                new_pt = pts[idx] + alpha * direction
                new_pt /= np.linalg.norm(new_pt)
                pts[idx] = new_pt
            new_rad = _covering_radius(pts)
            if new_rad > max_rad + 1e-8:
                # Revert again
                pts[i1] = v1
                pts[i2] = v2
                pts[i3] = v3
                break  # Stuck, stop optimizing
    
    return pts

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
    deadline = start_time + max_time - 10  # Safety margin
    
    best_pts = None
    best_radius = np.inf
    
    # Generate multiple initial configurations
    np.random.seed(42)
    configs = [_initial_fib()]
    
    # Add random configurations
    for seed in [123, 456, 789]:
        np.random.seed(seed)
        random_pts = np.random.randn(N, 3)
        random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
        configs.append(random_pts)
    
    for init_pts in configs:
        if time.time() > deadline - 30:
            break
        
        # Stage 1: Lloyd's algorithm for initial balancing
        lloyd_pts = _lloyd_iteration(init_pts, n_iter=5)
        
        # Evaluate after Lloyd
        rad = _covering_radius(lloyd_pts)
        if rad < best_radius:
            best_radius = rad
            best_pts = lloyd_pts.copy()
        
        # Stage 2: Facet-targeted optimization
        if time.time() > deadline - 20:
            break
        
        opt_pts = _facet_targeted_optimize(lloyd_pts, max_time_remaining=deadline - time.time())
        
        rad = _covering_radius(opt_pts)
        if rad < best_radius:
            best_radius = rad
            best_pts = opt_pts.copy()
    
    # Final stage: Local perturbation search
    if best_pts is None:
        best_pts = _initial_fib()
        best_radius = _covering_radius(best_pts)
    
    if time.time() < deadline - 5:
        # Try small perturbations on the worst facets
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            
            radii, simplices = _all_facet_radii(best_pts)
            max_rad = np.max(radii)
            max_idx = np.argmax(radii)
            worst_simplex = simplices[max_idx]
            
            i1, i2, i3 = worst_simplex[0], worst_simplex[1], worst_simplex[2]
            v1, v2, v3 = best_pts[i1], best_pts[i2], best_pts[i3]
            
            # Try random perturbations
            improved = False
            for trial in range(20):
                if time.time() > deadline - 5:
                    break
                delta = np.random.randn(3) * 0.02
                new_pt = best_pts[i1] + delta
                new_pt /= np.linalg.norm(new_pt)
                new_pts = best_pts.copy()
                new_pts[i1] = new_pt
                new_rad = _covering_radius(new_pts)
                if new_rad < best_radius - 1e-10:
                    best_pts = new_pts
                    best_radius = new_rad
                    improved = True
                    break
            
            if not improved:
                break
    
    # Ensure no coincident points and final normalization
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-8:
                best_pts[j] += np.random.randn(3) * 0.001
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
