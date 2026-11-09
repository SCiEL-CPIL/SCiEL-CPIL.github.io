import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _compute_facet_data(points):
    """
    Compute convex hull facets and their circumradii.
    Returns: (all_radii, circumcenters, simplices, valid_mask)
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None, None

    simplices = hull.simplices
    v1 = points[simplices[:, 0]]
    v2 = points[simplices[:, 1]]
    v3 = points[simplices[:, 2]]

    n_vec = np.cross(v2 - v1, v3 - v1)
    norms = np.linalg.norm(n_vec, axis=1)
    valid = norms > 1e-10
    if not np.any(valid):
        return None, None, None, None

    idx_valid = np.where(valid)[0]
    n_valid = n_vec[idx_valid]
    norms_valid = norms[idx_valid]
    v1_valid = v1[idx_valid]
    v2_valid = v2[idx_valid]
    v3_valid = v3[idx_valid]

    c_circ = n_valid / norms_valid[:, None]

    centroids = (v1_valid + v2_valid + v3_valid) / 3.0
    dots = np.sum(c_circ * centroids, axis=1)
    flip = dots < 0
    c_circ[flip] = -c_circ[flip]

    cos_theta = np.sum(c_circ * v1_valid, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)

    return radii, c_circ, simplices[idx_valid], valid

def _covering_radius(points):
    """Return the covering radius in radians."""
    radii, _, _, _ = _compute_facet_data(points)
    if radii is None:
        return np.pi
    return np.max(radii)

def _worst_facet_move(points, alpha=0.05):
    """
    Move the worst facet's vertices toward its circumcenter.
    Returns new points.
    """
    radii, c_circ, simplices, valid = _compute_facet_data(points)
    if radii is None:
        return points.copy()

    worst_idx = np.argmax(radii)
    worst_facet = simplices[worst_idx]
    worst_center = c_circ[worst_idx]

    new_pts = points.copy()
    for k in range(3):
        vi_idx = worst_facet[k]
        vi = new_pts[vi_idx]
        # Move toward circumcenter
        p_new = vi + alpha * (worst_center - vi)
        n_new = np.linalg.norm(p_new)
        if n_new > 1e-10:
            new_pts[vi_idx] = p_new / n_new

    return new_pts

def _multi_facet_move(points, alpha=0.02, top_k=3):
    """
    Move vertices of the top-k worst facets toward their circumcenters.
    Returns new points.
    """
    radii, c_circ, simplices, valid = _compute_facet_data(points)
    if radii is None:
        return points.copy()

    new_pts = points.copy()
    # Get indices of top-k worst facets
    top_indices = np.argsort(radii)[-top_k:]
    # Process from worst to least worst
    for idx in reversed(top_indices):
        facet = simplices[idx]
        center = c_circ[idx]
        for k in range(3):
            vi_idx = facet[k]
            vi = new_pts[vi_idx]
            p_new = vi + alpha * (center - vi)
            n_new = np.linalg.norm(p_new)
            if n_new > 1e-10:
                new_pts[vi_idx] = p_new / n_new

    return new_pts

def _smooth_gradient_step(points, beta=50.0, learning_rate=0.01):
    """
    Compute a gradient step using soft-max smoothing.
    Returns new points after the step.
    """
    radii, c_circ, simplices, valid = _compute_facet_data(points)
    if radii is None:
        return points.copy()

    max_r = np.max(radii)
    shifted = radii - max_r
    weights = np.exp(beta * shifted)
    total_w = np.sum(weights)
    weights /= total_w

    grad = np.zeros_like(points)
    sin_theta = np.sqrt(np.maximum(1.0 - np.cos(radii)**2, 1e-12))
    cos_theta = np.cos(radii)

    for i in range(len(radii)):
        w = weights[i]
        if w < 1e-8:
            continue
        s = sin_theta[i]
        if s < 1e-10:
            continue
        c = c_circ[i]
        ct = cos_theta[i]
        for k in range(3):
            vi = points[simplices[i, k]]
            g = -(c - vi * ct) / s
            grad[simplices[i, k]] += w * g

    # Project gradient onto tangent space
    radial = np.sum(grad * points, axis=1, keepdims=True)
    grad -= radial * points

    new_points = points - learning_rate * grad
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    new_points /= norms

    return new_points

def _local_optimize(points, max_iters=500, deadline=None):
    """
    Hybrid local optimization:
    - Primary: move worst facet vertices toward circumcenter
    - Secondary: smooth gradient step when progress stalls
    - Adaptive step sizes
    """
    best_pts = points.copy()
    best_cr = _covering_radius(best_pts)
    
    alpha_worst = 0.08
    alpha_multi = 0.03
    lr_smooth = 0.02
    stall_count = 0
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        current_cr = _covering_radius(points)
        
        # Try worst facet move first
        new_pts = _worst_facet_move(points, alpha_worst)
        new_cr = _covering_radius(new_pts)
        
        if new_cr < current_cr - 1e-10:
            points = new_pts
            stall_count = 0
            # Decay step size slightly
            alpha_worst *= 0.999
            alpha_multi *= 0.999
        else:
            # Try multi-facet move
            new_pts2 = _multi_facet_move(points, alpha_multi, top_k=3)
            new_cr2 = _covering_radius(new_pts2)
            
            if new_cr2 < current_cr - 1e-10:
                points = new_pts2
                stall_count = 0
                alpha_multi *= 0.999
            else:
                # Try smooth gradient step
                new_pts3 = _smooth_gradient_step(points, beta=50.0, learning_rate=lr_smooth)
                new_cr3 = _covering_radius(new_pts3)
                
                if new_cr3 < current_cr - 1e-10:
                    points = new_pts3
                    stall_count = 0
                    lr_smooth *= 0.99
                else:
                    # Stalled - perturb and reduce steps
                    stall_count += 1
                    alpha_worst *= 0.9
                    alpha_multi *= 0.9
                    lr_smooth *= 0.9
                    
                    if stall_count > 30:
                        # Add small random perturbation to escape local minimum
                        n_perturb = np.random.randint(1, 4)
                        for _ in range(n_perturb):
                            idx = np.random.randint(N)
                            noise = np.random.randn(3) * 0.01
                            p_new = points[idx] + noise
                            n_new = np.linalg.norm(p_new)
                            if n_new > 1e-10:
                                points[idx] = p_new / n_new
                        stall_count = 0
                        alpha_worst = 0.05
                        alpha_multi = 0.02
                        lr_smooth = 0.01
                    
                    if alpha_worst < 1e-8 and alpha_multi < 1e-8 and lr_smooth < 1e-8:
                        break
        
        # Track best
        cr = _covering_radius(points)
        if cr < best_cr:
            best_cr = cr
            best_pts = points.copy()
    
    return best_pts, best_cr

def _equalize_radii(points, max_iters=300, deadline=None):
    """
    Try to equalize all facet circumradii by moving vertices
    of facets with above-average radius toward their circumcenters.
    """
    best_pts = points.copy()
    best_cr = _covering_radius(best_pts)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        radii, c_circ, simplices, valid = _compute_facet_data(points)
        if radii is None:
            break
        
        current_cr = np.max(radii)
        if current_cr < best_cr:
            best_cr = current_cr
            best_pts = points.copy()
        
        mean_r = np.mean(radii)
        std_r = np.std(radii)
        
        if std_r < 1e-6:
            break
        
        # Find facets with radius above mean
        above_mean = radii > mean_r
        if not np.any(above_mean):
            break
        
        # Move vertices of above-mean facets toward circumcenters
        new_pts = points.copy()
        n_above = np.sum(above_mean)
        alpha = 0.01 * min(1.0, std_r / (mean_r + 1e-10))
        
        for i in range(len(radii)):
            if not above_mean[i]:
                continue
            facet = simplices[i]
            center = c_circ[i]
            for k in range(3):
                vi_idx = facet[k]
                vi = new_pts[vi_idx]
                p_new = vi + alpha * (center - vi)
                n_new = np.linalg.norm(p_new)
                if n_new > 1e-10:
                    new_pts[vi_idx] = p_new / n_new
        
        new_cr = _covering_radius(new_pts)
        if new_cr < current_cr:
            points = new_pts
        else:
            # If it got worse, revert and try smaller step
            alpha *= 0.5
            if alpha < 1e-8:
                break
    
    return best_pts, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    np.random.seed(42)

    best_points = None
    best_radius = np.pi

    def update_best(pts):
        nonlocal best_points, best_radius
        cr = _covering_radius(pts)
        if cr < best_radius:
            best_radius = cr
            best_points = pts.copy()
        return cr

    # Phase 1: Multiple diverse initializations
    initializations = [
        _fibonacci_sphere(N),
        _random_sphere(N, seed=1),
        _random_sphere(N, seed=2),
        _random_sphere(N, seed=3),
        _fibonacci_sphere(N) + np.random.randn(N, 3) * 0.05,
        _random_sphere(N, seed=4),
        _random_sphere(N, seed=5),
        _random_sphere(N, seed=6),
    ]

    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 50:
            break

        norms = np.linalg.norm(init_pts, axis=1, keepdims=True)
        init_pts = init_pts / np.where(norms < 1e-10, 1.0, norms)

        cr = update_best(init_pts)

        # Local optimization
        opt_pts, opt_cr = _local_optimize(init_pts, max_iters=400, deadline=deadline - 45)
        update_best(opt_pts)

        # Equalization pass
        if time.time() < deadline - 40:
            eq_pts, eq_cr = _equalize_radii(opt_pts, max_iters=200, deadline=deadline - 38)
            update_best(eq_pts)

    # Phase 2: Focused refinement on best solution
    if best_points is not None and time.time() < deadline - 25:
        pts = best_points.copy()
        
        # Multiple rounds of local optimization with decreasing step sizes
        for round_idx in range(3):
            if time.time() > deadline - 15:
                break
            opt_pts, opt_cr = _local_optimize(pts, max_iters=300, deadline=deadline - 15)
            update_best(opt_pts)
            pts = opt_pts
            
            # Equalization
            if time.time() < deadline - 12:
                eq_pts, eq_cr = _equalize_radii(pts, max_iters=150, deadline=deadline - 10)
                update_best(eq_pts)
                pts = eq_pts

    # Phase 3: Final high-precision polish
    if best_points is not None and time.time() < deadline - 8:
        pts = best_points.copy()
        
        # Very fine-tuned worst facet moves
        alpha = 0.01
        for it in range(300):
            if time.time() > deadline - 3:
                break
            
            new_pts = _worst_facet_move(pts, alpha)
            new_cr = _covering_radius(new_pts)
            
            if new_cr < _covering_radius(pts):
                pts = new_pts
                update_best(pts)
                alpha *= 0.995
            else:
                # Try multi-facet with tiny step
                new_pts2 = _multi_facet_move(pts, alpha * 0.5, top_k=2)
                new_cr2 = _covering_radius(new_pts2)
                if new_cr2 < _covering_radius(pts):
                    pts = new_pts2
                    update_best(pts)
                    alpha *= 0.995
                else:
                    alpha *= 0.9
                    if alpha < 1e-10:
                        break

    # Final validation and normalization
    if best_points is None:
        best_points = _fibonacci_sphere(N)

    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)

    # Ensure no coincident points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-6:
                axis = np.cross(best_points[i], best_points[j])
                if np.linalg.norm(axis) < 1e-10:
                    axis = np.array([1.0, 0.0, 0.0])
                axis /= np.linalg.norm(axis)
                theta = 0.01
                c, s = np.cos(theta), np.sin(theta)
                R = c * np.eye(3) + (1-c) * np.outer(axis, axis) + s * np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                best_points[j] = R @ best_points[j]
                best_points[j] /= np.linalg.norm(best_points[j])

    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)

    return best_points
