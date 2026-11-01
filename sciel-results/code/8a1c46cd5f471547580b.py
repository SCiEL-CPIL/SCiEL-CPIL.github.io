import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _spherical_circumcenters(hull, points):
    """
    Compute circumcenters and radii for all facets of the convex hull.
    Returns:
        centers: (M, 3) array of circumcenters (unit vectors)
        radii: (M,) array of angular radii (radians)
    """
    facets = hull.simplices
    centers = np.zeros((len(facets), 3))
    radii = np.zeros(len(facets))
    
    for i, face in enumerate(facets):
        v0 = points[face[0]]
        v1 = points[face[1]]
        v2 = points[face[2]]
        
        # Normal to the plane containing v0, v1, v2
        # This vector is equidistant (dot product) from v0, v1, v2
        n = np.cross(v1 - v0, v2 - v1)
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            # Degenerate facet, skip or handle
            centers[i] = v0
            radii[i] = 0
            continue
        
        center = n / norm
        
        # Ensure center is on the "outside" of the hull relative to the origin?
        # The circumcenter of the spherical Voronoi cell is the point on the sphere
        # that is the center of the largest empty circle.
        # The vector n points to one of the two possible circumcenters.
        # We need the one that is "far" from the interior of the hull?
        # Actually, for a convex hull on the sphere, the facet is on the boundary.
        # The circumcenter of the corresponding spherical triangle is the one
        # that lies in the direction of the outward normal of the facet?
        # Let's check dot product with the hull normal.
        
        # Hull normal for this face
        # hull.equations[i] gives [a,b,c,d] such that a*x+b*y+c*z+d=0
        # The normal is (a,b,c).
        # For a convex hull containing the origin (which it does, as points are on sphere),
        # the equation is a*x+b*y+c*z = d with d < 0? Or d > 0?
        # scipy.spatial.ConvexHull equates to: normal . x + offset = 0.
        # The normal points outwards.
        
        # Let's just check which of center or -center is "closer" to the facet in terms of being the center of the cap.
        # The cap is defined by the plane through v0,v1,v2.
        # The center of the cap is the intersection of the line through origin and the plane normal? No.
        # The circumcenter is the point on the sphere equidistant from v0,v1,v2.
        # There are two such points: center and -center.
        # One is inside the tetrahedron formed by origin and facet, the other is outside.
        # The Voronoi vertex (circumcenter) is the one that is NOT inside the hull?
        # Actually, the Voronoi cell is the set of points on the sphere closer to v_i than others.
        # The vertex of the Voronoi cell is the circumcenter of the triangle v0,v1,v2.
        # This point must be on the same side of the plane v0,v1,v2 as the origin?
        # No, the origin is inside the hull. The facet is on the boundary.
        # The circumcenter of the spherical triangle is the point on the sphere that is the center of the circle passing through v0,v1,v2.
        # If the triangle is "small", the circumcenter is close to the centroid.
        # Let's pick the one that has a larger dot product with the outward normal of the hull facet.
        
        eq = hull.equations[i]
        hull_normal = eq[:3]
        
        # If dot(center, hull_normal) > 0, center is in the direction of the outward normal.
        # The circumcenter of the Voronoi cell should be "outside" the hull, i.e., in the direction of the outward normal.
        if np.dot(center, hull_normal) < 0:
            center = -center
            
        centers[i] = center
        # Angular radius
        radii[i] = np.arccos(np.clip(np.dot(center, v0), -1.0, 1.0))
        
    return centers, radii

def _covering_radius(points):
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi # Fallback for invalid hull
    centers, radii = _spherical_circumcenters(hull, points)
    return np.max(radii)

def _optimize(points, max_iters=500, timeout=10):
    start_time = time.time()
    points = points.copy()
    step_size = 0.1
    min_step = 1e-6
    
    for i in range(max_iters):
        if time.time() - start_time > timeout:
            break
            
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        centers, radii = _spherical_circumcenters(hull, points)
        max_r = np.max(radii)
        
        if max_r < 1e-6:
            break
            
        # Identify worst facets
        # Use a threshold to include facets close to the max
        threshold = max_r - 0.01 
        worst_indices = np.where(radii > threshold)[0]
        
        if len(worst_indices) == 0:
            break
            
        # Accumulate gradients
        grads = np.zeros_like(points)
        
        for idx in worst_indices:
            face = hull.simplices[idx]
            center = centers[idx]
            r = radii[idx]
            
            # Weight based on how much worse it is
            # Simple weight: 1.0
            # Or proportional to r
            w = 1.0
            
            for j in face:
                v = points[j]
                # Move v towards center
                # Direction: center - v
                # We want to reduce the distance from center to v.
                # Moving v towards center reduces the chord distance, thus reducing angular distance.
                dir_vec = center - v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    grads[j] += w * dir_vec / norm_dir
                    
        # Normalize gradients
        grad_norms = np.linalg.norm(grads, axis=1, keepdims=True)
        grad_norms[grad_norms == 0] = 1
        grads /= grad_norms
        
        # Update points
        points += step_size * grads
        
        # Re-normalize to sphere
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Decay step size
        step_size *= 0.99
        
        if step_size < min_step:
            break
            
    return points, _covering_radius(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Leave 10s margin
    
    best_points = None
    best_radius = np.pi
    
    # Multi-start
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    starts = [init_fib]
    
    # 2. Random starts
    rng = np.random.default_rng(42)
    for _ in range(10):
        if time.time() > deadline:
            break
        p = rng.normal(size=(N, 3))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        starts.append(p)
        
    for start in starts:
        if time.time() > deadline:
            break
            
        # Allocate remaining time for this start
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        opt_points, radius = _optimize(start, timeout=remaining - 2)
        
        if radius < best_radius:
            best_radius = radius
            best_points = opt_points
            
    if best_points is None:
        return init_fib
        
    return best_points
