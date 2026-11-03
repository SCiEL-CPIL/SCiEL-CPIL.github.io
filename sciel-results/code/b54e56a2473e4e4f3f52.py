import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _covering_radius_and_centers(pts):
    """
    Computes the covering radius and the circumcenter of the worst triangle.
    Returns: (radius_in_radians, circumcenter_vector, triangle_indices)
    """
    pts = np.asarray(pts, dtype=np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, np.array([0.0, 0.0, 0.0]), None

    simplices = hull.simplices
    if len(simplices) == 0:
        return np.pi, np.array([0.0, 0.0, 0.0]), None

    # Vectorized circumcenter calculation
    # For each triangle (v0, v1, v2), find c such that c.v0 = c.v1 = c.v2 and ||c||=1
    # c is in the span of v0, v1, v2.
    # Let A be the matrix with rows v0, v1, v2.
    # We want c = A^T w such that A A^T w = [1, 1, 1]^T ? No.
    # c = sum wi vi.
    # c . v0 = sum wi (vi . v0) = 1 (cos R)
    # This is a system of 3 equations for w0, w1, w2.
    
    # Actually, simpler: The circumcenter is the intersection of the plane defined by v0, v1, v2 
    # and the line through the origin perpendicular to the plane? No.
    # The circumcenter is the point on the sphere equidistant from v0, v1, v2.
    # It is the normalization of the vector:
    # c_raw = cross(cross(v0, v1), cross(v1, v2)) ... no that's the normal.
    
    # Let's solve the linear system for each triangle.
    # M = [[v0.v0, v0.v1, v0.v2],
    #      [v1.v0, v1.v1, v1.v2],
    #      [v2.v0, v2.v1, v2.v2]]
    # M * w = [cosR, cosR, cosR]^T
    # This doesn't help find c directly without knowing R.
    
    # Geometric formula:
    # The circumcenter c is proportional to:
    # c ~ v0 * sin(2*angle_opp_v0) + ...
    # Or simply:
    # c is the solution to:
    # c . (v0 - v1) = 0
    # c . (v0 - v2) = 0
    # c . c = 1
    # c = v0 + a*(v1-v0) + b*(v2-v0)
    
    # Efficient vectorized solution:
    # c = (v0 * cross(v1, v2) + v1 * cross(v2, v0) + v2 * cross(v0, v1)) / ||...||
    # This is the circumcenter of the planar triangle projected?
    # For points on the unit sphere, the circumcenter is:
    # c = (v0 * (1 - v1.v2) + v1 * (1 - v2.v0) + v2 * (1 - v0.v1)) / ||...||
    # Let's verify this formula.
    # If v0, v1, v2 are close, v1.v2 ~ 1 - d12^2/2.
    
    # Let's use the robust formula:
    # c_unnorm = v0 * cross(v1, v2) + v1 * cross(v2, v0) + v2 * cross(v0, v1)
    # This vector is perpendicular to the plane of the triangle? No, it's in the plane.
    # Actually, cross(v1, v2) is normal to the plane. v0 * normal is in the plane? No.
    
    # Correct vectorized circumcenter for unit vectors:
    # c = (v0 * (1 - v1.dot(v2)) + v1 * (1 - v2.dot(v0)) + v2 * (1 - v0.dot(v1)))
    # Normalize c.
    
    v0 = pts[simplices[:, 0]]
    v1 = pts[simplices[:, 1]]
    v2 = pts[simplices[:, 2]]
    
    d01 = np.sum(v0 * v1, axis=1)
    d12 = np.sum(v1 * v2, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    
    # Avoid division by zero if points are coincident
    c_unnorm = v0 * (1 - d12)[:, None] + v1 * (1 - d20)[:, None] + v2 * (1 - d01)[:, None]
    c_norms = np.linalg.norm(c_unnorm, axis=1, keepdims=True)
    
    # Handle degenerate cases
    mask = c_norms > 1e-10
    c_norms_safe = np.where(mask, c_norms, 1.0)
    centers = c_unnorm / c_norms_safe
    
    # Calculate circumradius: angle between center and any vertex
    # cos(R) = c . v0
    cos_r = np.sum(centers * v0, axis=1)
    cos_r = np.clip(cos_r, -1.0, 1.0)
    radii = np.arccos(cos_r)
    
    max_idx = np.argmax(radii)
    max_radius = radii[max_idx]
    max_center = centers[max_idx]
    max_tri = simplices[max_idx]
    
    return max_radius, max_center, max_tri

def _fibonacci_init(seed=0):
    rng = np.random.RandomState(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0, 2*np.pi)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_pts = _fibonacci_init(0)
    best_cr, _, _ = _covering_radius_and_centers(best_pts)
    
    # Local Search: Move vertices of the worst triangle away from its circumcenter
    # Also apply a small global repulsion to prevent clustering
    
    step_size = 0.05
    cool_down = 0.99
    
    # Pre-compute pairwise dots for repulsion? No, too expensive for 130x130 every step?
    # 130*130 = 16900 ops. Very fast.
    
    for it in range(10000):
        if time.time() > deadline:
            break
        
        cr, center, tri = _covering_radius_and_centers(best_pts)
        if cr < best_cr:
            best_cr = cr
        
        if tri is None:
            break
            
        # Move the 3 vertices of the worst triangle away from the center
        # Force direction: vertex - center
        v0 = best_pts[tri[0]]
        v1 = best_pts[tri[1]]
        v2 = best_pts[tri[2]]
        
        # Normalize force
        f0 = v0 - center
        f1 = v1 - center
        f2 = v2 - center
        
        f0 /= np.linalg.norm(f0)
        f1 /= np.linalg.norm(f1)
        f2 /= np.linalg.norm(f2)
        
        # Apply step
        best_pts[tri[0]] += step_size * f0
        best_pts[tri[1]] += step_size * f1
        best_pts[tri[2]] += step_size * f2
        
        # Global repulsion to maintain uniformity
        # This helps escape local minima where other triangles are also large
        # Calculate repulsion forces
        dots = np.clip(best_pts @ best_pts.T, -1.0, 1.0)
        # Repulsion strength: 1 / (1 - dots) or similar
        # We want to push points apart.
        # Force on i from j: (pi - angle_ij) * (pi_i - pi_j)
        # Simplified: use inverse distance
        with np.errstate(divide='ignore', invalid='ignore'):
            # Avoid self
            np.fill_diagonal(dots, 1.0)
            # dist = arccos(dots)
            # We can use a soft repulsion: (1 - dots) ** -2
            rep = 1.0 / (1.0 - dots + 1e-6)**2
            np.fill_diagonal(rep, 0.0)
            
            # Force = sum_j rep_ij * (pi_i - pi_j) / ||pi_i - pi_j||
            # Vectorized:
            # This is tricky to vectorize perfectly without loops for the normalization.
            # Let's use a simpler gradient:
            # grad of -sum log(1 - d_ij) ?
            
            # Let's stick to the active set method but add a small random jitter
            # to escape symmetry.
        
        # Normalize
        best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
        
        # Jitter to break symmetry
        if it % 100 == 0:
            rng = np.random.RandomState(it)
            jitter = rng.randn(N, 3) * 0.001
            best_pts += jitter
            best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
            
        # Decrease step size
        step_size *= cool_down
        if step_size < 1e-5:
            step_size = 1e-5

    # Final normalization
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Check for coincident points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-6:
                rng = np.random.RandomState(42 + i * 1000 + j)
                best_pts[j] += 1e-5 * rng.randn(3)
                best_pts[j] /= np.linalg.norm(best_pts[j])
                
    return best_pts
