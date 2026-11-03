"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
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

def _facet_circumcenters(pts, hull):
    """Compute circumcenters and circumradii for all facets."""
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
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            cr = max(d_ab, d_bc, d_ca) / 2.0
            cc = (a + b + c) / 3.0
            cc = cc / np.linalg.norm(cc)
        else:
            u = v / norm_v
            n = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                continue
            n = n / norm_n
            if np.dot(n, a) < 0:
                n = -n
            if np.dot(u, n) > 0:
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
        
    return np.array(crs), np.array(ccs), simplices

def _force_based_relax(pts, target_cr, max_iter=200, time_limit=10.0):
    """
    Force-based relaxation: points repel each other and are attracted to
    fill gaps (facets with large circumradius).
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        crs, ccs, simplices = _facet_circumcenters(pts, hull)
        
        if len(crs) == 0:
            break
            
        max_cr = np.max(crs)
        
        # Forces: repulsion between all points + attraction to facet circumcenters
        forces = np.zeros((n, 3))
        
        # Repulsion: each point pushes away from others (soft repulsion)
        # Use a cutoff to avoid O(n^2) for large n, but n=50 is fine
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 0.5:  # Only repel when close
                    f = 0.1 / (d + 1e-6)
                    direction = (pts[i] - pts[j]) / (d + 1e-6)
                    forces[i] += f * direction
                    forces[j] -= f * direction
        
        # Attraction: points are pulled toward circumcenters of large facets
        beta = 5.0
        weights = np.exp(beta * (crs - max_cr))
        
        for idx, (simplex, cc, cr) in enumerate(zip(simplices, ccs, crs)):
            w = weights[idx]
            for pi in simplex:
                # Attraction toward cc, but only if it reduces the max cr
                direction = cc - pts[pi]
                direction = direction / (np.linalg.norm(direction) + 1e-6)
                forces[pi] += w * direction * 0.05
        
        # Project forces to tangent space and apply
        for i in range(n):
            forces[i] -= np.dot(forces[i], pts[i]) * pts[i]
        
        pts = pts + 0.1 * forces
        pts = _normalize(pts)
        
    return pts

def _polish(pts, time_limit=5.0):
    """
    High-precision polish: iteratively move points to reduce the max circumradius.
    Uses a smooth approximation and local search.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    start_time = time.time()
    
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    for it in range(500):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        crs, ccs, simplices = _facet_circumcenters(pts, hull)
        
        if len(crs) == 0:
            break
            
        max_cr = np.max(crs)
        
        # Find the worst facet and move its vertices slightly
        worst_idx = np.argmax(crs)
        worst_simplex = simplices[worst_idx]
        worst_cc = ccs[worst_idx]
        
        # Move vertices of worst facet toward circumcenter
        for pi in worst_simplex:
            direction = worst_cc - pts[pi]
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            pts[pi] += 0.02 * direction
        
        # Also apply mild repulsion to avoid coincident points
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 0.1:
                    direction = (pts[i] - pts[j]) / (d + 1e-6)
                    pts[i] += 0.01 * direction
                    pts[j] -= 0.01 * direction
        
        pts = _normalize(pts)
        
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
        else:
            # Revert if no improvement
            pts = best_pts.copy()
            # Try smaller step
            if 0.02 < 0.001:
                continue
            else:
                break
                
    return best_pts

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
        elif restart_count == 2:
            # Try a different initial configuration
            rng = np.random.RandomState(123)
            pts = rng.randn(N, 3)
            pts = _normalize(pts)
        else:
            # Perturb best solution
            noise_scale = 0.1 * (best_cr / 0.3)
            if noise_scale > 0.5:
                noise_scale = 0.5
            if noise_scale < 0.01:
                noise_scale = 0.01
            noise = noise_scale * np.random.randn(N, 3)
            pts = best_pts + noise
            pts = _normalize(pts)
            
        # 1. Lloyd's relaxation
        pts = _spherical_lloyd(pts, max_iter=80)
        
        # 2. Facet shrink
        pts = _optimize_facet_shrink(pts, max_iter=150, initial_step=0.3, decay=0.95)
        
        # 3. Force-based relaxation to fill gaps
        if time.time() < deadline - 20:
            target_cr = best_cr * 0.95
            pts = _force_based_relax(pts, target_cr, max_iter=100, time_limit=8.0)
        
        # 4. Final high-precision polish
        if time.time() < deadline - 10:
            pts = _polish(pts, time_limit=5.0)
            # Final facet shrink to clean up
            pts = _optimize_facet_shrink(pts, max_iter=50, initial_step=0.1, decay=0.9)
            
        if time.time() > deadline:
            break
            
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
            
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
