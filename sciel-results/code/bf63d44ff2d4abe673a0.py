import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_face_data(pts):
    """
    Compute circumradius, center, and vertex indices for all faces of the convex hull.
    """
    try:
        hull = ConvexHull(pts, qhull_options='QJ')
    except Exception:
        return None, None, None
        
    simplices = hull.simplices
    radii = np.zeros(len(simplices))
    centers = np.zeros((len(simplices), 3))
    
    for idx, tri in enumerate(simplices):
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        
        v1 = a - b
        v2 = b - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-10:
            center = (a + b + c) / 3.0
            norm_c = np.linalg.norm(center)
            if norm_c < 1e-10:
                radii[idx] = np.pi
                centers[idx] = np.array([0,0,0])
                continue
            center = center / norm_c
            cos_r = np.clip(np.dot(center, a), -1.0, 1.0)
            radii[idx] = np.arccos(cos_r)
            centers[idx] = center
        else:
            center = cross / norm_cross
            centroid = (a + b + c) / 3.0
            if np.dot(center, centroid) < 0:
                center = -center
            
            cos_r = np.clip(np.dot(center, a), -1.0, 1.0)
            radii[idx] = np.arccos(cos_r)
            centers[idx] = center
            
    return radii, centers, simplices

def _compute_covering_radius(pts):
    radii, _, _ = _compute_face_data(pts)
    if radii is None:
        return np.pi
    return np.max(radii)

def _smooth_gradient_step(pts, step_size, beta=50.0, grid_size=2000):
    """
    Compute gradient of the smooth covering radius using soft-min and take a Riemannian step.
    """
    i = np.arange(grid_size) + 0.5
    z = 1.0 - 2.0 * i / grid_size
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = grid @ pts.T
    dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
    
    angles = np.arccos(dots)
    
    shifted = -beta * angles
    max_shifted = np.max(shifted, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted - max_shifted)
    weights = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    
    sqrt_term = np.sqrt(np.maximum(1.0 - dots ** 2, 1e-12))
    inv_sqrt = 1.0 / sqrt_term
    
    grad = np.zeros((N, 3))
    for j in range(N):
        weighted = weights[:, j] * inv_sqrt[:, j]
        grad[j] = -np.sum(weighted[:, None] * grid, axis=0)
    
    for j in range(N):
        grad[j] = grad[j] - np.dot(grad[j], pts[j]) * pts[j]
    
    grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
    grad_norms = np.maximum(grad_norms, 1e-12)
    grad = grad / grad_norms
    
    new_pts = pts + step_size * grad
    new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
    
    return new_pts

def _dual_uniformization_step(pts, step_size, beta=10.0):
    """
    Perform a step that uniformizes the circumradii of the faces.
    """
    radii, centers, simplices = _compute_face_data(pts)
    if radii is None:
        return pts
        
    max_r = np.max(radii)
    weights = np.exp(beta * (radii - max_r))
    
    grad = np.zeros_like(pts)
    
    for k, tri in enumerate(simplices):
        w = weights[k]
        if w < 1e-8:
            continue
            
        center = centers[k]
        
        for idx in tri:
            v = pts[idx]
            dot_cv = np.dot(center, v)
            g = center - dot_cv * v
            grad[idx] += w * g
            
    new_pts = pts.copy()
    for i in range(N):
        g_norm = np.linalg.norm(grad[i])
        if g_norm > 1e-12:
            g = grad[i] / g_norm
            new_pts[i] = pts[i] + step_size * g
            new_pts[i] /= np.linalg.norm(new_pts[i])
            
    return new_pts

def _local_refine_combined(pts, deadline, max_iters=3000, initial_step=0.05, 
                           beta_smooth=50.0, beta_dual=10.0):
    """
    Local refinement combining smooth gradient and dual uniformization.
    Uses smooth gradient as primary, dual as fallback when smooth stalls.
    """
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(best_pts)
    
    step_size = initial_step
    no_improve_count = 0
    smooth_stall_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Primary: smooth gradient step
        new_pts_smooth = _smooth_gradient_step(pts, step_size, beta=beta_smooth, grid_size=2000)
        new_radius_smooth = _compute_covering_radius(new_pts_smooth)
        
        if new_radius_smooth < best_radius - 1e-12:
            best_radius = new_radius_smooth
            best_pts = new_pts_smooth.copy()
            pts = new_pts_smooth
            no_improve_count = 0
            smooth_stall_count = 0
            step_size *= 0.995
        else:
            # Try dual uniformization as fallback
            new_pts_dual = _dual_uniformization_step(pts, step_size, beta=beta_dual)
            new_radius_dual = _compute_covering_radius(new_pts_dual)
            
            if new_radius_dual < best_radius - 1e-12:
                best_radius = new_radius_dual
                best_pts = new_pts_dual.copy()
                pts = new_pts_dual
                no_improve_count = 0
                smooth_stall_count += 1
                step_size *= 0.995
            else:
                no_improve_count += 1
                smooth_stall_count += 1
                step_size *= 0.9
                
                if step_size < 1e-6 or no_improve_count > 50:
                    break
    
    return best_pts, best_radius

def _final_polish(pts, deadline):
    """
    High-precision final polish with tighter tolerances and more iterations.
    """
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(best_pts)
    
    step_size = 0.01
    for it in range(2000):
        if time.time() > deadline:
            break
            
        # Use smooth gradient with higher beta for precision
        new_pts = _smooth_gradient_step(pts, step_size, beta=100.0, grid_size=3000)
        new_radius = _compute_covering_radius(new_pts)
        
        if new_radius < best_radius - 1e-14:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
            step_size *= 0.99
        else:
            step_size *= 0.9
            if step_size < 1e-8:
                break
    
    return best_pts, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    np.random.seed(42)
    
    # Initial points: Fibonacci sphere
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_radius = _compute_covering_radius(best_points)
    
    # Phase 1: Combined refinement from Fibonacci start
    if time.time() < deadline - 40:
        refined_pts, refined_radius = _local_refine_combined(
            points.copy(), deadline - 40, max_iters=4000, initial_step=0.05
        )
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_points = refined_pts.copy()
    
    # Phase 2: Multiple restarts with different initializations
    n_restarts = 12
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        if restart == 0:
            pts = best_points + np.random.randn(N, 3) * 0.05
        elif restart == 1:
            pts = best_points + np.random.randn(N, 3) * 0.1
        elif restart == 2:
            pts = np.random.randn(N, 3)
        elif restart == 3:
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + np.pi / 4
            r = np.sqrt(1.0 - z * z)
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif restart == 4:
            random_pts = np.random.randn(N, 3)
            random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
            pts = 0.8 * best_points + 0.2 * random_pts
        elif restart == 5:
            # Fibonacci with different phase
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + np.pi / 3
            r = np.sqrt(1.0 - z * z)
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            pts = best_points + np.random.randn(N, 3) * (0.03 + 0.02 * restart)
            
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        if time.time() < deadline - 10:
            refined_pts, refined_radius = _local_refine_combined(
                pts, deadline - 10, max_iters=1500, initial_step=0.04
            )
            if refined_radius < best_radius:
                best_radius = refined_radius
                best_points = refined_pts.copy()
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 3:
        polished_pts, polished_radius = _final_polish(best_points.copy(), deadline - 3)
        if polished_radius < best_radius:
            best_radius = polished_radius
            best_points = polished_pts.copy()
    
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
