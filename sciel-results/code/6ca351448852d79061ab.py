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
        n = n / np.linalg.norm(n)
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

def _jittered_fibonacci(n, seed=42, jitter=0.1):
    """Fibonacci lattice with random jitter to break symmetry."""
    np.random.seed(seed)
    pts = _fibonacci_points(n)
    noise = jitter * np.random.randn(n, 3)
    pts = pts + noise
    return _normalize(pts)

def _spherical_lloyd(pts, max_iter=200, tol=1e-9):
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

def _facet_circumcenter(pts, simplex):
    """Compute circumcenter and circumradius of a spherical triangle facet."""
    a = pts[simplex[0]]
    b = pts[simplex[1]]
    c = pts[simplex[2]]
    
    v = np.cross(a - b, b - c)
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-12:
        # Degenerate: use midpoint of longest edge
        d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
        d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
        d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
        if d_ab >= d_bc and d_ab >= d_ca:
            cc = (a + b) / np.linalg.norm(a + b)
        elif d_bc >= d_ab and d_bc >= d_ca:
            cc = (b + c) / np.linalg.norm(b + c)
        else:
            cc = (c + a) / np.linalg.norm(c + a)
        cr = max(d_ab, d_bc, d_ca) / 2.0
        return cc, cr
    
    u = v / norm_v
    
    n_face = np.cross(b - a, c - a)
    n_face = n_face / np.linalg.norm(n_face)
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
    
    return cc, cr

def _gradient_descent_exact(pts, max_iter=500, lr=0.02, tol=1e-10):
    """
    Gradient descent on the exact covering radius using analytic gradients.
    The covering radius is max over facets of circumradius.
    We use the subgradient of the max function.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Find all facets and their circumradii
        n_facets = len(hull.simplices)
        crs = np.zeros(n_facets)
        ccs = [None] * n_facets
        
        for fi, simplex in enumerate(hull.simplices):
            cc, cr = _facet_circumcenter(pts, simplex)
            ccs[fi] = cc
            crs[fi] = cr
        
        max_cr = np.max(crs)
        
        if max_cr < 1e-8:
            break
        
        # Find active facets (within tolerance of max)
        active = np.where(crs > max_cr - 1e-8)[0]
        
        # Compute subgradient: average gradient contribution from active facets
        grad = np.zeros_like(pts)
        
        for fi in active:
            simplex = hull.simplices[fi]
            cc = ccs[fi]
            
            # For each vertex of this facet, compute gradient of circumradius w.r.t. that point
            # d(cr)/d(p_i) where cr = arccos(dot(cc, p_i)) for the farthest vertex
            # Actually, cr = max of distances from cc to each vertex
            # The gradient of arccos(t) w.r.t. t is -1/sqrt(1-t^2)
            # So d(cr)/d(p_i) = -1/sqrt(1-dot(cc,p_i)^2) * dot(cc, dp_i/dp_i)
            # But we need to be more careful: cr is the max over the three distances
            
            for vi, idx in enumerate(simplex):
                p = pts[idx]
                dot_val = np.clip(np.dot(cc, p), -1, 1)
                # If this vertex is the farthest, its gradient contributes
                d_cc_p = np.arccos(dot_val)
                
                # Check if this is the farthest vertex
                is_farthest = True
                for other_vi in range(3):
                    if other_vi == vi:
                        continue
                    other_idx = simplex[other_vi]
                    other_p = pts[other_idx]
                    other_dot = np.clip(np.dot(cc, other_p), -1, 1)
                    other_d = np.arccos(other_dot)
                    if other_d > d_cc_p + 1e-10:
                        is_farthest = False
                        break
                
                if is_farthest:
                    # Gradient of arccos(dot(cc, p)) w.r.t. p
                    # = -cc / sqrt(1 - dot(cc,p)^2)
                    denom = np.sqrt(max(1e-15, 1.0 - dot_val * dot_val))
                    grad[idx] += -cc / denom
        
        # Normalize the subgradient (average over active facets)
        n_active = len(active)
        if n_active > 0:
            grad /= n_active
        
        # Project onto tangent space
        for i in range(n):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        # Adaptive step size with line search
        step = lr / max(1.0, grad_norm)
        
        # Try a few step sizes
        best_step = step
        best_cr_new = max_cr
        for factor in [1.0, 0.5, 0.25, 0.1]:
            trial = pts - factor * step * grad
            trial = _normalize(trial)
            cr_trial = _covering_radius(trial)
            if cr_trial < best_cr_new:
                best_cr_new = cr_trial
                best_step = factor * step
        
        pts = _normalize(pts - best_step * grad)
        
        # Reduce learning rate
        lr *= 0.99
        
        if best_cr_new < max_cr - 1e-12:
            max_cr = best_cr_new
        else:
            # No improvement, reduce step
            lr *= 0.9
            
    return pts

def _high_precision_polish(pts, max_iter=200, tol=1e-12):
    """
    High-precision polish: iteratively adjust points to minimize the worst facet circumradius.
    Uses exact local optimization on the active facet.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Find worst facet
        max_cr = 0.0
        worst_facet = None
        worst_cc = None
        
        for simplex in hull.simplices:
            cc, cr = _facet_circumcenter(pts, simplex)
            if cr > max_cr:
                max_cr = cr
                worst_facet = simplex
                worst_cc = cc
        
        if worst_facet is None or max_cr < tol:
            break
        
        # Move the three vertices of the worst facet toward the circumcenter
        # with a small adaptive step
        step_size = 0.1 * max_cr
        
        for idx in worst_facet:
            p = pts[idx]
            vec = worst_cc - p
            vec_tan = vec - np.dot(vec, p) * p
            norm_vec = np.linalg.norm(vec_tan)
            if norm_vec > 1e-12:
                pts[idx] = p + step_size * (vec_tan / norm_vec)
        
        pts = _normalize(pts)
        
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    np.random.seed(42)
    
    best_pts = _jittered_fibonacci(N, seed=42, jitter=0.05)
    best_cr = _covering_radius(best_pts)
    
    n_restarts = 6
    
    for restart in range(n_restarts):
        if time.time() > deadline - 25:
            break
        
        if restart == 0:
            pts = _jittered_fibonacci(N, seed=42, jitter=0.05)
        elif restart == 1:
            pts = _fibonacci_points(N)
        elif restart == 2:
            vecs = np.random.randn(N, 3)
            pts = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        else:
            noise_scale = 0.15 / (1 + (restart - 3) * 0.2)
            noise = noise_scale * np.random.randn(N, 3)
            pts = best_pts + noise
            pts = _normalize(pts)
        
        # Stage 1: Lloyd's for global structure
        pts = _spherical_lloyd(pts, max_iter=100)
        
        # Stage 2: Exact gradient descent
        if time.time() < deadline - 20:
            pts = _gradient_descent_exact(pts, max_iter=300, lr=0.05, tol=1e-10)
        
        # Stage 3: High-precision polish
        if time.time() < deadline - 10:
            pts = _high_precision_polish(pts, max_iter=150, tol=1e-12)
        
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
