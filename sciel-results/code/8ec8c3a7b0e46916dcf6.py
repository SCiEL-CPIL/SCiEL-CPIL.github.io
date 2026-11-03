"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import math

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in radians using the convex hull approach."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius_grid(pts)
    
    max_cr = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        v = np.cross(a - b, b - c)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            cr = max(d_ab, d_bc, d_ca) / 2.0
            max_cr = max(max_cr, cr)
            continue
        
        u = v / norm_v
        
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            continue
        n = n / norm_n
        if np.dot(n, a) < 0:
            n = -n
            
        if np.dot(u, n) > 0:
            circum_center = u
        else:
            circum_center = -u
            
        d_a = np.arccos(np.clip(np.dot(circum_center, a), -1, 1))
        d_b = np.arccos(np.clip(np.dot(circum_center, b), -1, 1))
        d_c = np.arccos(np.clip(np.dot(circum_center, c), -1, 1))
        cr = max(d_a, d_b, d_c)
        
        max_cr = max(max_cr, cr)
        
    return max_cr

def _covering_radius_grid(pts, n_grid=100):
    """Fallback covering radius computation using a spherical grid."""
    pts = _normalize(pts)
    thetas = np.linspace(0, np.pi, n_grid)
    phis = np.linspace(0, 2 * np.pi, n_grid)
    THETA, PHI = np.meshgrid(thetas, phis)
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid = _normalize(grid)
    
    dots = grid @ pts.T
    dots = np.clip(dots, -1, 1)
    min_dots = np.max(dots, axis=1)
    min_angles = np.arccos(min_dots)
    return np.max(min_angles)

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _spherical_lloyd(pts, max_iter=1000, tol=1e-8):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    Moves each point to the centroid of its Voronoi cell.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        incident_facets = [[] for _ in range(n)]
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            normal = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-12:
                continue
            normal = normal / norm_n
            
            if np.dot(normal, a) < 0:
                normal = -normal
                
            for idx in simplex:
                incident_facets[idx].append(normal)
                
        new_pts = np.zeros_like(pts)
        max_move = 0.0
        
        for i in range(n):
            if len(incident_facets[i]) == 0:
                new_pts[i] = pts[i]
                continue
                
            centroid = np.sum(incident_facets[i], axis=0)
            norm_c = np.linalg.norm(centroid)
            
            if norm_c < 1e-10:
                new_pts[i] = pts[i]
            else:
                new_pts[i] = centroid / norm_c
                
            move = np.linalg.norm(new_pts[i] - pts[i])
            max_move = max(max_move, move)
            
        pts = _normalize(new_pts)
        
        if max_move < tol:
            break
            
    return pts

def _optimize_facet_shrink(pts, max_iter=1000, initial_step=0.5, decay=0.995, beta_start=2.0, beta_end=20.0):
    """
    Optimizes the configuration by moving each point towards the weighted
    average of the circumcenters of its incident facets.
    Uses adaptive step size and increasing beta to focus on worst facets.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    step_size = initial_step
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        weights_sum = np.zeros(n)
        vec_sum = np.zeros((n, 3))
        
        max_cr_global = 0.0
        facet_data = []

        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            v = np.cross(a - b, b - c)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-12:
                continue
            u = v / norm_v
            
            n_face = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n_face)
            if norm_n < 1e-12:
                continue
            n_face = n_face / norm_n
            if np.dot(n_face, a) < 0:
                n_face = -n_face
                
            if np.dot(u, n_face) > 0:
                cc = u
            else:
                cc = -u
                
            d_a = np.arccos(np.clip(np.dot(cc, a), -1, 1))
            d_b = np.arccos(np.clip(np.dot(cc, b), -1, 1))
            d_c = np.arccos(np.clip(np.dot(cc, c), -1, 1))
            cr = max(d_a, d_b, d_c)
            
            max_cr_global = max(max_cr_global, cr)
            facet_data.append((simplex, cc, cr))
            
        if max_cr_global < 1e-6:
            break

        progress = it / max_iter
        beta = beta_start + (beta_end - beta_start) * progress
        
        for simplex, cc, cr in facet_data:
            w = math.exp(beta * (cr - max_cr_global))
            
            for idx in simplex:
                weights_sum[idx] += w
                vec_sum[idx] += w * cc
                
        moved = False
        for i in range(n):
            if weights_sum[i] > 1e-10:
                target = vec_sum[i] / weights_sum[i]
                norm_t = np.linalg.norm(target)
                if norm_t > 1e-10:
                    target = target / norm_t
                    
                    vec = target - pts[i]
                    vec_tan = vec - np.dot(vec, pts[i]) * pts[i]
                    norm_vec = np.linalg.norm(vec_tan)
                    
                    if norm_vec > 1e-10:
                        pts[i] = pts[i] + step_size * vec_tan
                        moved = True
                        
        if not moved:
            break
            
        pts = _normalize(pts)
        
        step_size *= decay
        
        if step_size < 1e-8:
            break

    return pts

def _get_facet_data(pts):
    """Compute facet circumcenters and circumradii."""
    n = len(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, None, np.inf
    
    crs = []
    ccs = []
    simplices = []
    
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        v = np.cross(a - b, b - c)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            continue
        u = v / norm_v
        
        n_face = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n_face)
        if norm_n < 1e-12:
            continue
        n_face = n_face / norm_n
        if np.dot(n_face, a) < 0:
            n_face = -n_face
            
        if np.dot(u, n_face) > 0:
            cc = u
        else:
            cc = -u
            
        d_a = np.arccos(np.clip(np.dot(cc, a), -1, 1))
        d_b = np.arccos(np.clip(np.dot(cc, b), -1, 1))
        d_c = np.arccos(np.clip(np.dot(cc, c), -1, 1))
        cr = max(d_a, d_b, d_c)
        
        crs.append(cr)
        ccs.append(cc)
        simplices.append(simplex)
        
    if not crs:
        return None, None, None, np.inf
        
    return simplices, ccs, np.array(crs), np.max(crs)

def _polish(pts, max_iter=300, tol=1e-12, max_time=60.0):
    """
    High-precision polish: minimize the soft-max of facet circumradii
    using gradient descent with backtracking line search and adaptive beta.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    start_time = time.time()
    
    def get_obj_and_grad(p, beta):
        simplices, ccs, crs, max_cr = _get_facet_data(p)
        if simplices is None:
            return np.inf, np.zeros((n, 3))
            
        # Soft-max objective using log-sum-exp for numerical stability
        shifted = crs - max_cr
        log_sum_exp = max_cr + (1.0 / beta) * np.log(np.sum(np.exp(beta * shifted)))
        
        weights = np.exp(beta * shifted)
        sum_weights = np.sum(weights)
        if sum_weights < 1e-15:
            return log_sum_exp, np.zeros((n, 3))
        weights /= sum_weights
        
        grad = np.zeros((n, 3))
        for simplex, cc, w in zip(simplices, ccs, weights):
            for idx in simplex:
                vec = cc - p[idx]
                vec_tan = vec - np.dot(vec, p[idx]) * p[idx]
                grad[idx] += w * vec_tan
                
        return log_sum_exp, grad

    # Start with moderate beta, increase over time
    beta = 10.0
    step_size = 0.08
    
    current_obj, current_grad = get_obj_and_grad(pts, beta)
    
    for it in range(max_iter):
        if time.time() - start_time > max_time:
            break
            
        # Increase beta gradually to sharpen the max
        beta = 10.0 + 40.0 * (it / max_iter)
        
        if np.linalg.norm(current_grad) < tol:
            break
            
        # Backtracking line search
        alpha = step_size
        found = False
        for _ in range(30):
            new_pts = pts + alpha * current_grad
            new_pts = _normalize(new_pts)
            
            new_obj, new_grad = get_obj_and_grad(new_pts, beta)
            
            if new_obj < current_obj - 1e-14:
                pts = new_pts
                current_obj = new_obj
                current_grad = new_grad
                found = True
                break
            else:
                alpha *= 0.5
                
        if not found:
            step_size *= 0.5
            if step_size < 1e-8:
                break
        else:
            # Slightly increase step size if we made progress
            step_size = min(step_size * 1.1, 0.15)

    return pts

def _final_polish(pts, max_time=20.0):
    """
    Final high-precision polish with very tight tolerance and high beta.
    Uses a direct min-max approach by focusing only on the worst facets.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    start_time = time.time()
    
    def get_worst_facet_grad(p):
        simplices, ccs, crs, max_cr = _get_facet_data(p)
        if simplices is None:
            return np.inf, np.zeros((n, 3))
            
        # Find facets near the maximum
        threshold = max_cr - 1e-6
        worst_indices = [i for i, cr in enumerate(crs) if cr > threshold]
        
        if not worst_indices:
            worst_indices = [np.argmax(crs)]
            
        grad = np.zeros((n, 3))
        for idx in worst_indices:
            simplex = simplices[idx]
            cc = ccs[idx]
            for i in simplex:
                vec = cc - p[i]
                vec_tan = vec - np.dot(vec, p[i]) * p[i]
                grad[i] += vec_tan / len(worst_indices)
                
        return max_cr, grad

    step_size = 0.02
    current_obj, current_grad = get_worst_facet_grad(pts)
    
    for it in range(500):
        if time.time() - start_time > max_time:
            break
            
        if np.linalg.norm(current_grad) < 1e-14:
            break
            
        alpha = step_size
        found = False
        for _ in range(20):
            new_pts = pts + alpha * current_grad
            new_pts = _normalize(new_pts)
            
            new_obj, new_grad = get_worst_facet_grad(new_pts)
            
            if new_obj < current_obj - 1e-15:
                pts = new_pts
                current_obj = new_obj
                current_grad = new_grad
                found = True
                break
            else:
                alpha *= 0.5
                
        if not found:
            step_size *= 0.5
            if step_size < 1e-10:
                break

    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_pts = _fibonacci_points(N)
    best_cr = _covering_radius(best_pts)
    
    np.random.seed(42)
    
    restart_count = 0
    while time.time() < deadline - 25:  # Reserve 25s for final polish
        restart_count += 1
        
        if restart_count == 1:
            pts = _fibonacci_points(N)
        elif restart_count == 2:
            # Different initial: random with some structure
            pts = _fibonacci_points(N)
            pts[::3] += 0.1 * np.random.randn(len(pts[::3]), 3)
            pts = _normalize(pts)
        else:
            # Perturb best solution with varied noise scales
            noise_scales = [0.05, 0.1, 0.2, 0.3]
            scale = noise_scales[(restart_count - 3) % len(noise_scales)]
            noise = scale * np.random.randn(N, 3)
            pts = best_pts + noise
            pts = _normalize(pts)
            
        # 1. Lloyd's relaxation
        pts = _spherical_lloyd(pts, max_iter=60)
        
        # 2. Facet shrink
        pts = _optimize_facet_shrink(pts, max_iter=120, initial_step=0.4, decay=0.97)
        
        # 3. Polish if time allows
        if time.time() < deadline - 30:
            remaining = min(deadline - 30 - time.time(), 40.0)
            pts = _polish(pts, max_iter=200, tol=1e-12, max_time=remaining)
        
        if time.time() > deadline - 25:
            break
            
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    # Final high-precision polish on best solution
    if time.time() < deadline - 5:
        remaining = min(deadline - 5 - time.time(), 25.0)
        best_pts = _final_polish(best_pts, max_time=remaining)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
