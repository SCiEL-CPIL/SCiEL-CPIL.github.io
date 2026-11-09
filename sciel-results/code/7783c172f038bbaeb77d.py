"""
Covering the sphere with 130 equal caps.
Combines the structural idea of the current program (gradient polishing of hull facets)
with the refinement machinery from the inspiration (spherical triangle circumradius
computation, soft-max surrogate, and efficient projection).

Key improvements:
1. Use spherical geometry (spherical law of sines) for exact circumradius computation
   instead of planar circumcenter approximation.
2. Add a soft-max surrogate phase for smoother gradient-based optimization.
3. Better time management with explicit deadline guards.
4. Multi-restart with Fibonacci initialization variants.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _spherical_circumradius(p1, p2, p3):
    """
    Compute the circumradius of a spherical triangle with vertices p1, p2, p3
    (unit vectors) using the spherical law of sines.
    Returns the angle (in radians) from the circumcenter to each vertex.
    """
    # Side lengths (angles between pairs)
    a = np.arccos(np.clip(np.dot(p2, p3), -1.0, 1.0))
    b = np.arccos(np.clip(np.dot(p1, p3), -1.0, 1.0))
    c = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
    
    s = 0.5 * (a + b + c)
    
    # Spherical law of sines: sin(R) / sin(A) = sin(a) / (2 * sin(s) * sin(s-a) * sin(s-b) * sin(s-c))
    # More precisely: sin(R) = sin(a)*sin(b)*sin(c) / (2 * sin(s) * sin(s-a) * sin(s-b) * sin(s-c))
    # But the standard formula is:
    # sin(R) = sin(a) * sin(b) * sin(c) / (4 * sin(s) * sin(s-a) * sin(s-b) * sin(s-c))
    
    denom = 4.0 * np.sin(s) * np.sin(s - a) * np.sin(s - b) * np.sin(s - c)
    
    if abs(denom) < 1e-15:
        # Degenerate case: return the largest half-side
        return max(a, b, c) / 2.0
    
    sin_R = np.sin(a) * np.sin(b) * np.sin(c) / denom
    sin_R = np.clip(sin_R, 0.0, 1.0)
    R = np.arcsin(sin_R)
    
    # Handle the case where the circumcenter is on the "other side"
    # If R > pi/2, the circumcenter is antipodal; use pi - R? 
    # Actually for covering, we want the smaller cap, so R should be <= pi/2
    # But if the triangle is large, R could be > pi/2.
    # The circumradius for covering is the max distance to the nearest point,
    # so if R > pi/2, the antipodal point is closer.
    # But typically for well-distributed points, R < pi/2.
    
    return R

def _covering_radius(pts):
    """
    Compute the covering radius in radians using the convex hull.
    The covering radius is the maximum circumradius of the spherical Delaunay triangles,
    which correspond to the facets of the convex hull.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use a grid-based estimate
        return _covering_radius_grid(pts)
    
    max_rad = 0.0
    for simplex in hull.simplices:
        p1 = pts[simplex[0]]
        p2 = pts[simplex[1]]
        p3 = pts[simplex[2]]
        
        R = _spherical_circumradius(p1, p2, p3)
        if R > max_rad:
            max_rad = R
    
    return max_rad

def _covering_radius_grid(pts):
    """Fallback: estimate covering radius via grid sampling."""
    # Use a moderately fine grid
    n_lat = 40
    n_lon = 80
    lat = np.linspace(-np.pi/2 + 1e-6, np.pi/2 - 1e-6, n_lat)
    lon = np.linspace(0, 2*np.pi - 1e-6, n_lon)
    L, O = np.meshgrid(lat, lon)
    x = np.cos(L) * np.cos(O)
    y = np.cos(L) * np.sin(O)
    z = np.sin(L)
    grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    max_rad = 0.0
    # Process in chunks for memory efficiency
    chunk_size = 1000
    for i in range(0, len(grid_pts), chunk_size):
        chunk = grid_pts[i:i+chunk_size]
        # Compute min angular distance from each grid point to the nearest pts
        dots = np.clip(chunk @ pts.T, -1.0, 1.0)  # (chunk, N)
        angles = np.arccos(dots)
        min_angles = np.min(angles, axis=1)
        max_rad = max(max_rad, np.max(min_angles))
    
    return max_rad

def _get_worst_facet(pts):
    """
    Find the facet of the convex hull with the largest circumradius.
    Returns (worst_circumcenter, simplex_indices, radius) or (None, None, 0) on failure.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, 0.0
    
    max_rad = 0.0
    worst_c = None
    worst_simplex = None
    
    for simplex in hull.simplices:
        p1 = pts[simplex[0]]
        p2 = pts[simplex[1]]
        p3 = pts[simplex[2]]
        
        R = _spherical_circumradius(p1, p2, p3)
        
        if R > max_rad:
            max_rad = R
            # Compute the circumcenter direction
            # The circumcenter is in the direction of the normal to the plane through p1, p2, p3
            v1 = p2 - p1
            v2 = p3 - p1
            c = np.cross(v1, v2)
            norm_c = np.linalg.norm(c)
            if norm_c > 1e-12:
                c = c / norm_c
                # Choose the sign that points toward the "empty" side
                # For a convex hull facet, the outward normal points away from the origin
                # The circumcenter of the spherical triangle (the center of the empty cap)
                # should be on the same side as the outward normal
                # Check: dot(c, p1) should be positive for the "close" circumcenter
                if np.dot(c, p1) < 0:
                    c = -c
                worst_c = c
            else:
                worst_c = None
            worst_simplex = simplex
    
    return worst_c, worst_simplex, max_rad

def _soft_max_surrogate(pts, samples, beta):
    """
    Compute a smooth surrogate for the covering radius using soft-min/soft-max.
    
    For each sample point y, the distance to the nearest pt is min_j d(y, pt_j).
    The covering radius is max_y min_j d(y, pt_j).
    
    We approximate min_j with soft-min: softmin_j(d_j) = -1/beta * log(sum_j exp(-beta * d_j))
    And max_y with soft-max: softmax_y(v_y) = 1/beta * log(sum_y exp(beta * v_y))
    
    We use chord distance d = 1 - y·pt_j as a monotone proxy for angular distance.
    Actually, let's use the angular distance directly but it's non-differentiable at 0.
    Better: use d = arccos(y·pt_j) but clip to avoid issues.
    Or use squared chord distance for smoothness: d = 2(1 - y·pt_j).
    
    For gradient computation, we need to be careful.
    Let's use the chord-based distance: d_ij = 1 - samples[i] · pts[j]
    This is in [0, 2] and is monotone with angular distance.
    
    softmin_j(d_ij) = -1/beta * log(sum_j exp(-beta * d_ij))
    surrogate = 1/beta * log(sum_i exp(beta * softmin_i))
    
    Gradient w.r.t. pts[j]:
    d surrogate / d pts[j] = sum_i w_i * d(softmin_i)/d(pts[j])
    where w_i = exp(beta * softmin_i) / sum_k exp(beta * softmin_k)
    
    d(softmin_i)/d(pts[j]) = -1/beta * d/d(pts[j]) log(sum_k exp(-beta * d_ik))
    = -1/beta * [sum_k exp(-beta*d_ik) * (-beta) * d(d_ik)/d(pts[j])] / sum_k exp(-beta*d_ik)
    = sum_k exp(-beta*d_ik) * d(d_ik)/d(pts[j]) / sum_k exp(-beta*d_ik)
    
    d(d_ik)/d(pts[j]) = 0 if k != j, = -samples[i] if k == j
    So d(softmin_i)/d(pts[j]) = exp(-beta*d_ij) * (-samples[i]) / sum_k exp(-beta*d_ik)
    
    Therefore:
    grad_j = -sum_i w_i * exp(-beta*d_ij) / S_i * samples[i]
    where S_i = sum_k exp(-beta*d_ik)
    """
    M = len(samples)
    
    # Compute d_ij = 1 - samples[i] · pts[j], shape (M, N)
    dots = np.clip(samples @ pts.T, -1.0, 1.0)
    d = 1.0 - dots  # shape (M, N)
    
    # Soft-min over j for each i
    neg_beta_d = -beta * d  # (M, N)
    max_neg = np.max(neg_beta_d, axis=1, keepdims=True)  # (M, 1)
    exp_neg = np.exp(neg_beta_d - max_neg)  # (M, N)
    S = np.sum(exp_neg, axis=1)  # (M,)
    log_S = np.log(S)
    soft_min = -(max_neg.squeeze() + log_S) / beta  # (M,)
    
    # Soft-max over i
    beta_sm = beta
    beta_softmin = beta_sm * soft_min  # (M,)
    max_beta_sm = np.max(beta_softmin)
    exp_beta_sm = np.exp(beta_softmin - max_beta_sm)  # (M,)
    sum_exp = np.sum(exp_beta_sm)
    surrogate = (max_beta_sm + np.log(sum_exp)) / beta_sm
    
    # Weights for soft-max
    w = exp_beta_sm / sum_exp  # (M,)
    
    # Probabilities for soft-min
    probs = exp_neg / S[:, None]  # (M, N)
    
    # Gradient: grad[j] = -sum_i w[i] * probs[i,j] * samples[i]
    # = -(w[:, None] * probs).T @ samples
    weighted_probs = w[:, None] * probs  # (M, N)
    grad = -(weighted_probs.T @ samples)  # (N, 3)
    
    return surrogate, grad

def _fibonacci_sphere(n, seed=0):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0, 2 * np.pi)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=0):
    """Generate n random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _gradient_polish(pts, max_time, deadline):
    """
    Polish the configuration by iteratively moving the vertices of the worst facet
    toward the circumcenter to reduce the covering radius.
    """
    start_time = time.time()
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_rad = _covering_radius(best_pts)
    
    step_size = 0.02
    
    while time.time() < deadline - 2.0:
        c, simplex, rad = _get_worst_facet(pts)
        if c is None or simplex is None:
            break
        
        if rad <= best_rad + 1e-10:
            # No improvement possible from this facet; try a smaller step
            step_size *= 0.8
            if step_size < 1e-6:
                break
            continue
        
        p1 = pts[simplex[0]]
        p2 = pts[simplex[1]]
        p3 = pts[simplex[2]]
        
        # Compute gradients: move each vertex toward the circumcenter
        # The gradient of the circumradius w.r.t. each vertex points toward the circumcenter
        # We use a simple step toward c
        for idx, p in [(simplex[0], p1), (simplex[1], p2), (simplex[2], p3)]:
            # Move toward c
            pts[idx] = p + step_size * c
        
        pts = _normalize(pts)
        
        new_rad = _covering_radius(pts)
        if new_rad < best_rad - 1e-10:
            best_rad = new_rad
            best_pts = pts.copy()
            # Slightly increase step size if making progress
            step_size = min(0.05, step_size * 1.05)
        else:
            # Revert
            pts[simplex[0]] = p1
            pts[simplex[1]] = p2
            pts[simplex[2]] = p3
            pts = _normalize(pts)
            step_size *= 0.85
            if step_size < 1e-7:
                break
    
    return best_pts, best_rad

def _soft_max_optimize(pts, max_time, deadline):
    """
    Optimize using the soft-max surrogate for smoother gradients.
    """
    start_time = time.time()
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_rad = _covering_radius(best_pts)
    
    # Generate sample points on the sphere for the surrogate
    rng = np.random.RandomState(42)
    n_samples = 2000
    samples = rng.randn(n_samples, 3)
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    
    beta = 20.0
    lr = 0.002
    
    for iteration in range(5000):
        if time.time() > deadline - 5.0:
            break
        
        surrogate, grad = _soft_max_surrogate(pts, samples, beta)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm[grad_norm < 1e-12] = 1.0
        grad = grad / grad_norm
        
        # Update
        pts = pts - lr * grad
        pts = _normalize(pts)
        
        # Occasionally check the true covering radius
        if iteration % 100 == 0:
            current_rad = _covering_radius(pts)
            if current_rad < best_rad - 1e-10:
                best_rad = current_rad
                best_pts = pts.copy()
                # Decrease learning rate slightly
                lr *= 0.95
            else:
                # Revert to best and try smaller step
                pts = best_pts.copy()
                lr *= 0.8
                if lr < 1e-6:
                    break
    
    return best_pts, best_rad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before the 150s kill
    
    best_pts = None
    best_rad = np.inf
    
    # Try multiple initializations
    inits = []
    for seed in range(5):
        inits.append(_fibonacci_sphere(N, seed))
    for seed in range(3):
        inits.append(_random_sphere(N, seed + 100))
    
    for init in inits:
        if time.time() > deadline - 30.0:
            break
        
        pts = _normalize(init.copy())
        rad = _covering_radius(pts)
        
        if rad < best_rad:
            best_rad = rad
            best_pts = pts.copy()
        
        # Phase 1: Gradient polish (fast, exact)
        if time.time() < deadline - 20.0:
            pts1, rad1 = _gradient_polish(pts, max_time=15.0, deadline=deadline - 5.0)
            if rad1 < best_rad:
                best_rad = rad1
                best_pts = pts1.copy()
        
        # Phase 2: Soft-max optimization (smoother, may escape local minima)
        if time.time() < deadline - 15.0:
            pts2, rad2 = _soft_max_optimize(pts1, max_time=10.0, deadline=deadline - 5.0)
            if rad2 < best_rad:
                best_rad = rad2
                best_pts = pts2.copy()
    
    # Final polish on the best configuration
    if best_pts is not None and time.time() < deadline - 5.0:
        final_pts, final_rad = _gradient_polish(best_pts, max_time=8.0, deadline=deadline - 2.0)
        if final_rad < best_rad:
            best_rad = final_rad
            best_pts = final_pts
    
    if best_pts is None:
        # Should not happen, but fallback
        best_pts = _fibonacci_sphere(N, 0)
    
    return _normalize(best_pts)

# EVOLVE-BLOCK-END
