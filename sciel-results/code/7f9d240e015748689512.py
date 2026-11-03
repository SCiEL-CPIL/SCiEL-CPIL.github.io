"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import math
import random

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

def _polish(pts, max_iter=200, tol=1e-10):
    """
    High-precision polish: minimize the soft-max of facet circumradii
    using gradient descent with very small steps.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    step_size = 0.01
    beta = 50.0  # High beta for sharp max approximation
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Compute all facet circumradii
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
            break
        
        crs = np.array(crs)
        max_cr = np.max(crs)
        
        if max_cr < tol:
            break
        
        # Soft-max weights
        weights = np.exp(beta * (crs - max_cr))
        
        # Compute gradient: move each point away from weighted circumcenters
        grad = np.zeros((n, 3))
        for simplex, cc, w in zip(simplices, ccs, weights):
            for idx in simplex:
                # Move point towards circumcenter to reduce its circumradius
                vec = cc - pts[idx]
                vec_tan = vec - np.dot(vec, pts[idx]) * pts[idx]
                grad[idx] += w * vec_tan
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm[grad_norm < 1e-15] = 1.0
        grad = grad / grad_norm
        
        # Update
        max_move = 0.0
        for i in range(n):
            if np.linalg.norm(grad[i]) > 1e-10:
                pts[i] = pts[i] + step_size * grad[i]
                move = np.linalg.norm(grad[i]) * step_size
                max_move = max(max_move, move)
        
        pts = _normalize(pts)
        
        if max_move < tol:
            break
        
        # Slowly decay step size
        step_size *= 0.999

    return pts

def _simulated_annealing(pts, start_time, deadline, T_start=0.15, T_end=0.001, 
                          cooling=0.998, local_opt_iters=30):
    """
    Simulated annealing on the sphere.
    
    Moves:
    1. Random perturbation of 1-3 points (Gaussian noise on tangent space)
    2. Small local optimization (few Lloyd iterations)
    
    Acceptance: Metropolis criterion on covering radius.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    T = T_start
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    while time.time() < deadline:
        # Generate move: perturb 1-3 random points
        n_perturb = random.randint(1, 3)
        idxs = random.sample(range(n), n_perturb)
        
        new_pts = pts.copy()
        for idx in idxs:
            # Random tangent direction
            rand_vec = np.random.randn(3)
            tangent = rand_vec - np.dot(rand_vec, new_pts[idx]) * new_pts[idx]
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-10:
                continue
            tangent = tangent / tangent_norm
            
            # Random step size scaled by temperature
            step = np.random.uniform(-1, 1) * T * 10
            new_pts[idx] = new_pts[idx] + step * tangent
        
        new_pts = _normalize(new_pts)
        
        # Small local optimization to settle
        new_pts = _spherical_lloyd(new_pts, max_iter=local_opt_iters)
        
        new_cr = _covering_radius(new_pts)
        
        # Metropolis acceptance
        delta_cr = new_cr - best_cr
        if delta_cr < 0 or random.random() < math.exp(-delta_cr / T):
            pts = new_pts
            if new_cr < best_cr:
                best_cr = new_cr
                best_pts = new_pts.copy()
        
        # Cool down
        T *= cooling
        if T < T_end:
            T = T_end
        
        # Occasionally do a stronger local optimization
        if random.random() < 0.1 and time.time() < deadline - 15:
            pts = _optimize_facet_shrink(pts, max_iter=50, initial_step=0.3, decay=0.98)
            cr = _covering_radius(pts)
            if cr < best_cr:
                best_cr = cr
                best_pts = pts.copy()
    
    return best_pts, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    np.random.seed(42)
    random.seed(42)
    
    # Phase 1: Get a good initial solution with local optimization
    pts = _fibonacci_points(N)
    pts = _spherical_lloyd(pts, max_iter=100)
    pts = _optimize_facet_shrink(pts, max_iter=200, initial_step=0.4, decay=0.97)
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Phase 2: Simulated annealing to explore the landscape
    if time.time() < deadline - 30:
        sa_deadline = deadline - 20
        best_pts, best_cr = _simulated_annealing(
            best_pts, start_time, sa_deadline, 
            T_start=0.12, T_end=0.002, 
            cooling=0.9985, local_opt_iters=20
        )
    
    # Phase 3: Final polish
    if time.time() < deadline - 10:
        best_pts = _polish(best_pts, max_iter=150, tol=1e-11)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
