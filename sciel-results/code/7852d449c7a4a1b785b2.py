import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _compute_covering_radius(pts):
    """
    Computes the covering radius in radians.
    Returns (radius, worst_facet_indices)
    """
    try:
        hull = ConvexHull(pts)
        facets = hull.simplices
        max_cr = 0.0
        worst_idx = None
        for i, j, k in facets:
            p_i = pts[i]
            p_j = pts[j]
            p_k = pts[k]
            
            # Circumcenter direction
            c = p_i + p_j + p_k
            norm_c = np.linalg.norm(c)
            if norm_c < 1e-10:
                continue
            c = c / norm_c
            
            # Circumradius
            dot = np.dot(c, p_i)
            dot = np.clip(dot, -1.0, 1.0)
            theta = np.arccos(dot)
            
            if theta > max_cr:
                max_cr = theta
                worst_idx = (i, j, k)
                
        return max_cr, worst_idx
    except Exception:
        return np.pi, None

def _fibonacci_sphere(n):
    """Generate a Fibonacci sphere distribution."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(points)

def _move_point_on_sphere(p, direction, step):
    """
    Move point p on the sphere by step size in the direction of 'direction'.
    Uses exponential map approximation: p_new = cos(step)*p + sin(step)*d
    where d is the normalized tangent direction.
    """
    # Project direction to tangent space
    dot = np.dot(direction, p)
    tangent = direction - dot * p
    norm_t = np.linalg.norm(tangent)
    if norm_t < 1e-10:
        return p
    d = tangent / norm_t
    
    # Exponential map
    p_new = np.cos(step) * p + np.sin(step) * d
    return p_new / np.linalg.norm(p_new)

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s kill
    
    # Strategy:
    # 1. Start with Fibonacci sphere (good global structure).
    # 2. Iteratively identify the worst facet (largest circumradius).
    # 3. Calculate the subgradient: to minimize the circumradius of a facet (i,j,k),
    #    we should move the vertices away from the circumcenter.
    #    Actually, the circumradius R is determined by the distance from origin to the plane? No, it's on the sphere.
    #    The circumcenter C is (pi+pj+pk)/|pi+pj+pk|.
    #    The radius is arccos(C . pi).
    #    To decrease the radius, we need to increase C . pi.
    #    This means moving pi towards C?
    #    Let's check: if pi moves towards C, the angle decreases.
    #    Wait, if pi, pj, pk form a triangle, the circumcenter is the center of the circle passing through them.
    #    If we move pi towards C, we are "pulling" the vertex towards the center of the hole.
    #    This reduces the size of the hole.
    #    
    #    Let's verify the gradient direction.
    #    f(p) = arccos(C . p). df/dp = -1/sqrt(1-(C.p)^2) * C.
    #    To minimize f, we move in direction -df/dp = + C / sqrt(...).
    #    So we should move the vertices TOWARDS the circumcenter C.
    #
    #    However, moving one vertex towards C might increase the circumradius of adjacent facets.
    #    We will move all 3 vertices of the worst facet towards their common circumcenter.
    
    pts = _fibonacci_sphere(N)
    current_radius, worst_idx = _compute_covering_radius(pts)
    best_pts = pts.copy()
    best_radius = current_radius
    
    # Adaptive step size
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        current_radius, worst_idx = _compute_covering_radius(pts)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
        
        if worst_idx is None:
            break
            
        i, j, k = worst_idx
        p_i = pts[i]
        p_j = pts[j]
        p_k = pts[k]
        
        # Compute circumcenter of the worst facet
        c = p_i + p_j + p_k
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-10:
            # Degenerate case, pick a random point and move it
            idx = np.random.randint(N)
            direction = np.random.randn(3)
            pts[idx] = _move_point_on_sphere(pts[idx], direction, step_size)
            continue
        
        c = c / norm_c
        
        # Move vertices towards the circumcenter
        # We use a scaled step size. If the step is too large, it might cause issues or increase radius.
        # We will try a few step sizes or use a fixed one with backtracking if it fails.
        
        new_pts = pts.copy()
        
        # Move i towards c
        new_pts[i] = _move_point_on_sphere(p_i, c, step_size)
        # Move j towards c
        new_pts[j] = _move_point_on_sphere(p_j, c, step_size)
        # Move k towards c
        new_pts[k] = _move_point_on_sphere(p_k, c, step_size)
        
        new_radius, _ = _compute_covering_radius(new_pts)
        
        if new_radius < current_radius:
            pts = new_pts
            # Optionally increase step size slightly if improving
            step_size = min(step_size * 1.05, 0.2)
        else:
            # If not improving, reduce step size
            step_size *= 0.8
            if step_size < 1e-6:
                # If step size is too small, try a random kick or break
                # Let's try a random kick to escape local minimum
                idx = np.random.randint(N)
                direction = np.random.randn(3)
                pts[idx] = _move_point_on_sphere(pts[idx], direction, 0.05)
                step_size = 0.05
                
    return _normalize(best_pts)

if __name__ == "__main__":
    pts = construct()
    print(f"Shape: {pts.shape}")
    print(f"Norms: {np.linalg.norm(pts, axis=1)[:5]}")
    rad, _ = _compute_covering_radius(pts)
    print(f"Covering Radius: {rad * 180 / np.pi:.6f} degrees")
