"""
Strategy: Deterministic Worst-Facet Shrinkage with Jittered Fibonacci Initialization.

This approach differs from the seed program by:
1. Using a jittered Fibonacci spiral for initialization to break symmetries.
2. Implementing a specific "Worst-Facet Shrinkage" step that identifies the facet
   with the largest circumradius and moves its vertices directly toward the 
   circumcenter (the "empty" cap center) to shrink that specific facet.
3. Combining this with Lloyd's algorithm in a multi-stage pipeline.
4. Using diverse restarts to escape local minima.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius_and_facets(pts):
    """
    Compute the covering radius (in radians) and identify the critical facet.
    Returns (radius, facet_indices, circumcenter_direction).
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback if hull fails
        return _covering_radius_fallback(pts), None, None
    
    max_radius = 0.0
    best_facet = None
    best_center = None
    
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Circumcenter direction
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            continue
        
        x = cross / norm_cross
        
        # Angular distance from x to vertex a
        dot_a = np.dot(x, a)
        dot_a = np.clip(dot_a, -1.0, 1.0)
        r1 = np.arccos(dot_a)
        
        # The other direction
        dot_a_neg = -dot_a
        dot_a_neg = np.clip(dot_a_neg, -1.0, 1.0)
        r2 = np.arccos(dot_a_neg)
        
        # Take the larger circumradius (the one in the "empty" cap direction)
        candidate = max(r1, r2)
        if candidate > max_radius:
            max_radius = candidate
            best_facet = simplex
            # Determine which direction is the "empty" one
            # The outward normal of the facet defined by a,b,c in CCW order (from outside) 
            # is cross(b-a, c-a). 
            # ConvexHull simplices are ordered such that the normal points outward.
            # So x is the direction of the empty cap center.
            best_center = x
            
    # Check antipodes of each point
    min_dist_to_anti = 0.0
    for i in range(len(pts)):
        neg_pt = -pts[i]
        dots = pts @ neg_pt
        dots = np.clip(dots, -1.0, 1.0)
        min_d = np.arccos(np.min(dots))
        if min_d > max_radius:
            max_radius = min_d
            # In this case, the "facet" is effectively the antipode
            best_facet = None 
            best_center = neg_pt

    return max_radius, best_facet, best_center

def _covering_radius_fallback(pts):
    """Fallback covering radius computation via moderate sampling."""
    n_theta = 100
    n_phi = 200
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    max_radius = 0.0
    for theta in thetas:
        st = np.sin(theta)
        ct = np.cos(theta)
        y_batch = np.column_stack([
            st * np.cos(phis),
            st * np.sin(phis),
            np.full_like(phis, ct)
        ])
        dots = pts @ y_batch.T
        dots = np.clip(dots, -1.0, 1.0)
        min_dists = np.arccos(np.min(dots, axis=0))
        max_radius = max(max_radius, np.max(min_dists))
    return max_radius

def _objective(pts):
    """Compute the objective (covering radius in radians)."""
    r, _, _ = _covering_radius_and_facets(pts)
    return r

def _spherical_lloyd_iteration(pts):
    """
    Perform one iteration of Spherical Lloyd's algorithm.
    Moves each point to the spherical centroid of its Voronoi cell.
    """
    pts = _to_sphere(pts.copy())
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts

    # Map each point to the facets that contain it
    point_to_facets = {i: [] for i in range(N)}
    
    for idx, simplex in enumerate(hull.simplices):
        for i in simplex:
            point_to_facets[i].append(idx)

    new_pts = np.zeros_like(pts)
    
    for i in range(N):
        facets = point_to_facets[i]
        if not facets:
            new_pts[i] = pts[i]
            continue
            
        # Collect the circumcenters of the adjacent facets.
        centers = []
        for f_idx in facets:
            simplex = hull.simplices[f_idx]
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            v1 = a - b
            v2 = a - c
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            if norm_cross > 1e-12:
                x = cross / norm_cross
                centers.append(x)
        
        if not centers:
            new_pts[i] = pts[i]
            continue
        
        centers = np.array(centers)
        
        # The spherical centroid of a set of points on the sphere is the normalized sum
        centroid = np.sum(centers, axis=0)
        norm_centroid = np.linalg.norm(centroid)
        if norm_centroid > 1e-12:
            new_pts[i] = centroid / norm_centroid
        else:
            new_pts[i] = pts[i]
            
    return _to_sphere(new_pts)

def _local_optimize_lloyd(pts, max_time):
    """
    Local optimization using Spherical Lloyd's algorithm.
    """
    pts = _to_sphere(pts.copy())
    current_obj = _objective(pts)
    best_pts = pts.copy()
    best_obj = current_obj
    
    start = time.time()
    no_improve = 0
    dampening = 1.0
    
    while time.time() - start < max_time:
        new_pts = _spherical_lloyd_iteration(pts)
        
        # Dampening to prevent oscillation
        pts_damped = (1 - dampening) * pts + dampening * new_pts
        pts_damped = _to_sphere(pts_damped)
        
        new_obj = _objective(pts_damped)
        
        if new_obj < current_obj:
            pts = pts_damped
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = pts.copy()
            no_improve = 0
            dampening = min(dampening * 1.05, 1.0)
        else:
            no_improve += 1
            dampening *= 0.9
            if no_improve > 20:
                break
    
    return best_pts, best_obj

def _worst_facet_shrinkage(pts, max_time, step_size=0.05):
    """
    Iteratively identify the worst facet and shrink it by moving its vertices
    toward the circumcenter.
    """
    pts = _to_sphere(pts.copy())
    current_obj = _objective(pts)
    best_pts = pts.copy()
    best_obj = current_obj
    
    start = time.time()
    
    while time.time() - start < max_time:
        r, facet, center = _covering_radius_and_facets(pts)
        
        if facet is None:
            # Bottleneck is an antipode, hard to fix with local facet shrinkage
            # Try moving the isolated point toward the mean of others
            if center is not None:
                dots = pts @ (-center)
                k = np.argmax(dots)
                mask = np.ones(N, dtype=bool)
                mask[k] = False
                mean_other = np.sum(pts[mask], axis=0)
                norm_mean = np.linalg.norm(mean_other)
                if norm_mean > 1e-12:
                    mean_other = mean_other / norm_mean
                    proj = np.dot(mean_other, pts[k])
                    tangential = mean_other - proj * pts[k]
                    norm_t = np.linalg.norm(tangential)
                    if norm_t > 1e-12:
                        new_pts = pts.copy()
                        new_pts[k] = pts[k] + step_size * (tangential / norm_t)
                        new_pts = _to_sphere(new_pts)
                        new_obj = _objective(new_pts)
                        if new_obj < current_obj:
                            pts = new_pts
                            current_obj = new_obj
                            if current_obj < best_obj:
                                best_obj = current_obj
                                best_pts = pts.copy()
                        else:
                            step_size *= 0.9
                            if step_size < 0.001:
                                break
            else:
                break
        else:
            # Move the three vertices of the worst facet toward the center
            new_pts = pts.copy()
            for i in facet:
                proj = np.dot(center, pts[i])
                tangential = center - proj * pts[i]
                norm_t = np.linalg.norm(tangential)
                if norm_t > 1e-12:
                    new_pts[i] = pts[i] + step_size * (tangential / norm_t)
            
            new_pts = _to_sphere(new_pts)
            new_obj = _objective(new_pts)
            
            if new_obj < current_obj:
                pts = new_pts
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_pts = pts.copy()
                # Slightly increase step size if improving rapidly
                step_size = min(step_size * 1.05, 0.1)
            else:
                step_size *= 0.9
                if step_size < 0.001:
                    break
    
    return best_pts, best_obj

def _jittered_fibonacci_init(jitter_scale=0.1, seed=42):
    """
    Generate a jittered Fibonacci spiral configuration.
    """
    np.random.seed(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add jitter to break symmetries
    jitter = np.random.randn(N, 3) * jitter_scale
    pts = pts + jitter
    pts = _to_sphere(pts)
    
    return pts

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    # Start with Jittered Fibonacci spiral initial configuration
    pts = _jittered_fibonacci_init(jitter_scale=0.1, seed=42)
    
    best_pts = pts.copy()
    best_obj = _objective(pts)
    
    deadline = _DEADLINE
    remaining = deadline - time.time()
    
    # Phase 1: Lloyd's Algorithm for global structure
    if remaining > 10:
        pts, obj = _local_optimize_lloyd(pts, max_time=min(remaining - 10, 20))
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
    
    # Phase 2: Worst-Facet Shrinkage for local precision
    remaining = deadline - time.time()
    if remaining > 10:
        pts, obj = _worst_facet_shrinkage(best_pts, max_time=min(remaining - 10, 20))
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
    
    # Phase 3: Diverse Restarts
    remaining = deadline - time.time()
    np.random.seed(42)
    for restart in range(8):
        if time.time() > deadline - 15:
            break
        
        # Create a perturbed version
        jitter_scale = 0.2 + 0.1 * restart
        perturbed = _jittered_fibonacci_init(jitter_scale=jitter_scale, seed=42 + restart)
        
        # Short Lloyd
        if remaining > 20:
            pts, obj = _local_optimize_lloyd(perturbed, max_time=min(remaining - 20, 10))
            if obj < best_obj:
                best_obj = obj
                best_pts = pts.copy()
        
        # Short Worst-Facet Shrinkage
        remaining = deadline - time.time()
        if remaining > 10:
            pts, obj = _worst_facet_shrinkage(best_pts, max_time=min(remaining - 10, 10))
            if obj < best_obj:
                best_obj = obj
                best_pts = pts.copy()
        
        remaining = deadline - time.time()
    
    # Final refinement
    remaining = deadline - time.time()
    if remaining > 5:
        pts, obj = _worst_facet_shrinkage(best_pts, max_time=min(remaining - 5, 5))
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
    
    best_pts = _to_sphere(best_pts)
    
    # Validate
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(np.abs(dots)) > 0.999999:
        best_pts = _to_sphere(best_pts + np.random.randn(N, 3) * 0.01)
        
    return best_pts
# EVOLVE-BLOCK-END
