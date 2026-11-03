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
        # Fallback: grid
        return _covering_radius_grid(pts)
    
    max_cr = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute circumcenter of spherical triangle a,b,c
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

        # Interpolate beta from beta_start to beta_end
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
        
        # Decay step size
        step_size *= decay
        
        if step_size < 1e-8:
            break

    return pts

def _soft_max_covering_radius(pts, beta=50.0):
    """
    Compute soft-max covering radius and its gradient w.r.t. point positions.
    Uses the convex hull to find facets and their circumradii.
    Returns (soft_max_cr, gradient_array) where gradient has shape (n, 3).
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: return a large value and zero gradient
        return 1.0, np.zeros((n, 3))
    
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
        return 1.0, np.zeros((n, 3))
    
    crs = np.array(crs)
    max_cr = np.max(crs)
    
    if max_cr < 1e-10:
        return max_cr, np.zeros((n, 3))
    
    # Soft-max: log(sum(exp(beta * crs))) / beta
    # For numerical stability, subtract max
    scaled = beta * (crs - max_cr)
    log_sum_exp = max_cr * beta + np.log(np.sum(np.exp(scaled)))
    soft_max_cr = log_sum_exp / beta
    
    # Weights for soft-max
    weights = np.exp(scaled) / np.sum(np.exp(scaled))
    
    # Compute gradient
    grad = np.zeros((n, 3))
    for simplex, cc, w in zip(simplices, ccs, weights):
        for idx in simplex:
            # The circumradius depends on the angle between cc and pts[idx]
            # d = arccos(dot(cc, p))
            # dd/dp = -cc / sqrt(1 - dot(cc,p)^2) in the tangent direction
            dot_val = np.dot(cc, pts[idx])
            dot_val = np.clip(dot_val, -1 + 1e-15, 1 - 1e-15)
            sin_d = np.sqrt(1 - dot_val * dot_val)
            if sin_d < 1e-15:
                continue
            # Gradient of arccos(x) w.r.t. x is -1/sqrt(1-x^2)
            # Gradient of dot(cc, p) w.r.t. p is cc
            # So gradient of d w.r.t. p is -cc / sin_d
            # But we want to MINIMIZE d, so we move in direction -grad_d = cc/sin_d
            # However, we need to project onto tangent space
            vec = cc / sin_d
            vec_tan = vec - np.dot(vec, pts[idx]) * pts[idx]
            grad[idx] += w * vec_tan
    
    return soft_max_cr, grad

def _polish(pts, deadline, tol=1e-10):
    """
    High-precision polish using L-BFGS-B on a smooth soft-max covering radius objective.
    Uses tangent-space parameterization.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    # Use tangent-space coordinates: represent each point as a perturbation
    # from a base point using local coordinates
    # For simplicity, use the points directly but project gradients to tangent space
    
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Try L-BFGS-B optimization
    try:
        from scipy.optimize import minimize
        
        def objective_and_grad(x):
            """Objective: soft-max covering radius. x is flattened (n,3) array."""
            p = x.reshape(n, 3)
            p = _normalize(p)
            val, grad = _soft_max_covering_radius(p, beta=50.0)
            # Project gradient to tangent space
            grad_proj = np.zeros_like(grad)
            for i in range(n):
                grad_proj[i] = grad[i] - np.dot(grad[i], p[i]) * p[i]
            return val, grad_proj.ravel()
        
        x0 = pts.ravel()
        
        # Set up bounds to keep points on unit sphere (loose bounds)
        bounds = [(-1.5, 1.5)] * (n * 3)
        
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={
                'maxiter': 50,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 20
            }
        )
        
        if result.success or result.fun < best_cr:
            new_pts = _normalize(result.x.reshape(n, 3))
            new_cr = _covering_radius(new_pts)
            if new_cr < best_cr:
                best_pts = new_pts
                best_cr = new_cr
        
    except Exception:
        pass
    
    # Final local polish with small steps if time allows
    if time.time() < deadline - 2:
        pts = best_pts
        step_size = 0.005
        beta = 100.0
        
        for it in range(50):
            if time.time() > deadline - 1:
                break
                
            val, grad = _soft_max_covering_radius(pts, beta=beta)
            
            if val < 1e-10:
                break
            
            # Project gradient to tangent space
            grad_proj = np.zeros_like(grad)
            for i in range(n):
                grad_proj[i] = grad[i] - np.dot(grad[i], pts[i]) * pts[i]
            
            # Normalize
            grad_norms = np.linalg.norm(grad_proj, axis=1, keepdims=True)
            grad_norms[grad_norms < 1e-15] = 1.0
            grad_proj = grad_proj / grad_norms
            
            # Update
            max_move = 0.0
            for i in range(n):
                if np.linalg.norm(grad_proj[i]) > 1e-10:
                    pts[i] = pts[i] + step_size * grad_proj[i]
                    move = np.linalg.norm(grad_proj[i]) * step_size
                    max_move = max(max_move, move)
            
            pts = _normalize(pts)
            
            if max_move < tol:
                break
            
            step_size *= 0.995
        
        final_cr = _covering_radius(pts)
        if final_cr < best_cr:
            best_pts = pts
            best_cr = final_cr
    
    return _normalize(best_pts)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    best_pts = _fibonacci_points(N)
    best_cr = _covering_radius(best_pts)
    
    np.random.seed(42)
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        if restart_count == 1:
            pts = _fibonacci_points(N)
        else:
            # Perturb best solution
            # Scale noise based on current best radius to ensure meaningful moves
            noise_scale = 0.15 * (best_cr / 0.3)
            if noise_scale > 0.4:
                noise_scale = 0.4
            if noise_scale < 0.02:
                noise_scale = 0.02
            noise = noise_scale * np.random.randn(N, 3)
            pts = best_pts + noise
            pts = _normalize(pts)
            
        # 1. Lloyd's relaxation to get a good local minimum for variance
        pts = _spherical_lloyd(pts, max_iter=80)
        
        # 2. Facet shrink to minimize covering radius
        pts = _optimize_facet_shrink(pts, max_iter=150, initial_step=0.4, decay=0.97)
        
        # 3. High-precision polish using L-BFGS-B on smooth surrogate
        if time.time() < deadline - 25:
            pts = _polish(pts, deadline=deadline - 5, tol=1e-10)
        
        if time.time() > deadline:
            break
            
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
            
    # Final polish on best solution if time allows
    if time.time() < deadline - 10:
        best_pts = _polish(best_pts, deadline=deadline - 2, tol=1e-11)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
