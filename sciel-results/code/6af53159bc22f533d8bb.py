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

def _compute_covering_metrics(points):
    """
    Returns (max_radius, worst_facet_indices, circumcenter, all_radii)
    max_radius in radians
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, None, None, np.array([])
    
    simplices = hull.simplices
    v1 = points[simplices[:, 0]]
    v2 = points[simplices[:, 1]]
    v3 = points[simplices[:, 2]]
    
    n = np.cross(v2 - v1, v3 - v1)
    norms = np.linalg.norm(n, axis=1)
    valid = norms > 1e-10
    if not np.any(valid):
        return np.pi, None, None, np.array([])
    
    idx_valid = np.where(valid)[0]
    n_valid = n[idx_valid]
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
    
    max_idx = np.argmax(radii)
    max_radius = radii[max_idx]
    worst_facet = simplices[idx_valid[max_idx]]
    worst_center = c_circ[max_idx]
    
    return max_radius, worst_facet, worst_center, radii

def _spherical_gradient_step(pts, step_size=0.01, grid_size=2000):
    """
    Compute gradient of the smooth covering radius and take a Riemannian gradient step.
    Uses a soft-max approximation to get a smooth objective.
    """
    M = grid_size
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Dots: (M, N)
    dots = grid @ pts.T
    
    # Soft-max over points for each grid point
    beta = 40.0
    shifted = beta * dots
    max_shifted = np.max(shifted, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted - max_shifted)
    weights = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    
    # Weighted dot product (smooth max)
    g = np.sum(weights * dots, axis=1)
    g = np.clip(g, -1.0 + 1e-12, 1.0 - 1e-12)
    d = np.arccos(g)
    
    # Soft-max over grid points
    beta2 = 40.0
    shifted2 = beta2 * d
    max_shifted2 = np.max(shifted2)
    exp_shifted2 = np.exp(shifted2 - max_shifted2)
    y_weights = exp_shifted2 / np.sum(exp_shifted2)
    
    # Gradient w.r.t. each point p_i
    # dE/dp_i = -sum_y y_weights[y] * weights[y,i] * grid[y] / sqrt(1-g[y]^2)
    sqrt_term = np.sqrt(np.maximum(1.0 - g**2, 1e-12))
    inv_sqrt = 1.0 / sqrt_term
    
    grad = np.zeros((N, 3))
    for i in range(N):
        weighted = y_weights * weights[:, i] * inv_sqrt
        grad[i] = -np.sum(weighted[:, None] * grid, axis=0)
    
    # Project gradient to tangent space
    for i in range(N):
        grad[i] = grad[i] - np.dot(grad[i], pts[i]) * pts[i]
    
    # Normalize gradient
    grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
    grad_norm = np.maximum(grad_norm, 1e-12)
    grad = grad / grad_norm
    
    # Riemannian gradient step
    new_pts = pts + step_size * grad
    new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
    
    return new_pts

def _local_refine(pts, max_iters=2000, step_size=0.03, decay=0.9995, deadline=None):
    """Local refinement: move worst facet vertices toward circumcenter."""
    current_radius, worst_facet, worst_center, _ = _compute_covering_metrics(pts)
    best_local = pts.copy()
    best_local_radius = current_radius
    
    alpha = step_size
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        current_radius, worst_facet, worst_center, _ = _compute_covering_metrics(pts)
        
        if current_radius < best_local_radius:
            best_local_radius = current_radius
            best_local = pts.copy()
        
        if worst_facet is None:
            break
        
        v1_idx, v2_idx, v3_idx = worst_facet
        v1, v2, v3 = pts[v1_idx], pts[v2_idx], pts[v3_idx]
        
        p1_new = v1 + alpha * (worst_center - v1)
        p2_new = v2 + alpha * (worst_center - v2)
        p3_new = v3 + alpha * (worst_center - v3)
        
        n1 = np.linalg.norm(p1_new)
        n2 = np.linalg.norm(p2_new)
        n3 = np.linalg.norm(p3_new)
        
        if n1 > 1e-10 and n2 > 1e-10 and n3 > 1e-10:
            pts[v1_idx] = p1_new / n1
            pts[v2_idx] = p2_new / n2
            pts[v3_idx] = p3_new / n3
        
        alpha *= decay
        if alpha < 1e-8:
            break
    
    return best_local, best_local_radius

def _gradient_descent(pts, max_iters=1000, step_size=0.01, decay=0.999, deadline=None):
    """Pure gradient descent on the smooth covering radius."""
    best = pts.copy()
    best_radius, _, _, _ = _compute_covering_metrics(best)
    
    alpha = step_size
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        new_pts = _spherical_gradient_step(pts, step_size=alpha, grid_size=2000)
        new_radius, _, _, _ = _compute_covering_metrics(new_pts)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best = new_pts.copy()
            pts = new_pts
        else:
            alpha *= 0.9
            if alpha < 1e-6:
                break
        
        alpha *= decay
    
    return best, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Initial configuration
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_radius, _, _, _ = _compute_covering_metrics(best_points)
    
    np.random.seed(42)
    
    # Phase 1: Rapid local refinement from Fibonacci
    best_points, best_radius = _local_refine(points, max_iters=3000, step_size=0.05, decay=0.9995, deadline=deadline)
    
    # Phase 2: Gradient descent from best
    if time.time() < deadline - 30:
        gd_pts, gd_radius = _gradient_descent(best_points, max_iters=500, step_size=0.01, decay=0.999, deadline=deadline)
        if gd_radius < best_radius:
            best_radius = gd_radius
            best_points = gd_pts.copy()
    
    # Phase 3: Multiple restarts with mixed strategies
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 40:
            break
        
        # Generate initial points
        if restart == 0:
            pts = best_points.copy()
        elif restart == 1:
            pts = _fibonacci_sphere(N)
            noise = np.random.randn(N, 3) * 0.1
            pts = pts + noise
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif restart == 2:
            # Random points
            pts = np.random.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif restart == 3:
            # Mix with best solution
            pts = np.random.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            mix = 0.5
            pts = (1 - mix) * pts + mix * best_points
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            # Perturbed best
            pts = best_points + np.random.randn(N, 3) * 0.05
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        current_radius, _, _, _ = _compute_covering_metrics(pts)
        
        # Simulated annealing with gradient steps
        temperature = 0.03
        cooling = 0.997
        min_temperature = 1e-5
        max_sa_iters = 2000
        stuck_counter = 0
        best_sa_radius = current_radius
        best_sa_pts = pts.copy()
        
        for it in range(max_sa_iters):
            if time.time() > deadline - 15:
                break
            
            # Choose a random point to move
            idx = np.random.randint(N)
            
            current_radius, worst_facet, worst_center, all_radii = _compute_covering_metrics(pts)
            
            if worst_facet is not None:
                v1_idx, v2_idx, v3_idx = worst_facet
                if idx in [v1_idx, v2_idx, v3_idx]:
                    target = worst_center
                    move = target - pts[idx]
                    move_norm = np.linalg.norm(move)
                    if move_norm > 1e-10:
                        move = move / move_norm * temperature
                else:
                    move = np.random.randn(3) * temperature
            else:
                move = np.random.randn(3) * temperature
            
            new_pt = pts[idx] + move
            new_pt_norm = np.linalg.norm(new_pt)
            if new_pt_norm < 1e-10:
                continue
            new_pt = new_pt / new_pt_norm
            
            old_pt = pts[idx].copy()
            pts[idx] = new_pt
            
            new_radius, _, _, _ = _compute_covering_metrics(pts)
            
            delta = new_radius - current_radius
            if delta < 0 or np.random.rand() < np.exp(-delta / max(temperature, 1e-10)):
                current_radius = new_radius
                if new_radius < best_sa_radius:
                    best_sa_radius = new_radius
                    best_sa_pts = pts.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                pts[idx] = old_pt
                stuck_counter += 1
            
            temperature *= cooling
            if temperature < min_temperature:
                break
            
            # If stuck, do a gradient step
            if stuck_counter > 100 and it % 30 == 0:
                pts = _spherical_gradient_step(pts, step_size=0.005, grid_size=1500)
                current_radius, _, _, _ = _compute_covering_metrics(pts)
                if current_radius < best_sa_radius:
                    best_sa_radius = current_radius
                    best_sa_pts = pts.copy()
                    stuck_counter = 0
        
        # Update global best
        if best_sa_radius < best_radius:
            best_radius = best_sa_radius
            best_points = best_sa_pts.copy()
        
        # Local refine the best from this restart
        if time.time() < deadline - 10:
            refined_pts, refined_radius = _local_refine(best_sa_pts, max_iters=500, step_size=0.02, decay=0.9995, deadline=deadline)
            if refined_radius < best_radius:
                best_radius = refined_radius
                best_points = refined_pts.copy()
    
    # Phase 4: Final gradient polish
    if time.time() < deadline - 10:
        gd_pts, gd_radius = _gradient_descent(best_points, max_iters=300, step_size=0.005, decay=0.9995, deadline=deadline)
        if gd_radius < best_radius:
            best_radius = gd_radius
            best_points = gd_pts.copy()
    
    # Phase 5: Final local polish
    if time.time() < deadline - 5:
        final_pts, final_radius = _local_refine(best_points, max_iters=2000, step_size=0.005, decay=0.9995, deadline=deadline)
        if final_radius < best_radius:
            best_radius = final_radius
            best_points = final_pts.copy()
    
    # Ensure final points are normalized and valid
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    return best_points
