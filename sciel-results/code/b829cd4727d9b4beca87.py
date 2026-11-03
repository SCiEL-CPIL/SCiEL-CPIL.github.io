"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import math
from scipy.optimize import minimize

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

def _random_points(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    pts = np.random.randn(n, 3)
    return _normalize(pts)

def _icosahedron_base():
    """Return 12 vertices of a regular icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    return _normalize(verts)

def _spherical_lloyd(pts, max_iter=100, tol=1e-8):
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

def _smooth_covering_radius(pts, beta=50.0):
    """
    Smooth surrogate for covering radius using log-sum-exp over facet circumradii.
    This provides a differentiable approximation to the max over facets.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius_grid(pts), np.zeros(len(pts))
    
    crs = []
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
            crs.append(cr)
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
            cc = u
        else:
            cc = -u
            
        d_a = np.arccos(np.clip(np.dot(cc, a), -1, 1))
        d_b = np.arccos(np.clip(np.dot(cc, b), -1, 1))
        d_c = np.arccos(np.clip(np.dot(cc, c), -1, 1))
        cr = max(d_a, d_b, d_c)
        crs.append(cr)
        
    if not crs:
        return _covering_radius_grid(pts), np.zeros(len(pts))
    
    crs = np.array(crs)
    max_cr = np.max(crs)
    
    # Log-sum-exp for smooth max
    shifted = crs - max_cr
    log_sum_exp = max_cr + np.log(np.sum(np.exp(beta * shifted)) / len(crs))
    
    # Compute gradient w.r.t. points (approximate via finite differences would be too slow)
    # Instead, we'll use the Nelder-Mead method which doesn't need gradients
    return log_sum_exp, np.zeros(len(pts))

def _optimize_with_nm(pts, max_iter=200, timeout=None, start_time=None):
    """
    Optimize using Nelder-Mead on the smooth surrogate objective.
    """
    if start_time is None:
        start_time = time.time()
    
    pts = _normalize(pts.copy())
    n = len(pts)
    
    def objective(x):
        if timeout is not None and time.time() - start_time > timeout:
            return 1e10  # Penalize timeout
        pts_try = _normalize(x.reshape(n, 3))
        try:
            val, _ = _smooth_covering_radius(pts_try, beta=30.0)
            return val
        except Exception:
            return 1e10
    
    # Use Nelder-Mead with adaptive settings
    result = minimize(objective, pts.flatten(), method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12,
                              'adaptive': True, 'initial_simplex_size': 0.1})
    
    if result.success or result.fun < 1e9:
        return _normalize(result.x.reshape(n, 3))
    return pts

def _facet_shrink_polish(pts, max_iter=500, initial_step=0.3, decay=0.99, beta_start=5.0, beta_end=50.0):
    """
    High-precision polish pass using weighted facet circumcenter attraction.
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
            
        if max_cr_global < 1e-8:
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
        
        if step_size < 1e-10:
            break

    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    best_pts = _fibonacci_points(N)
    best_cr = _covering_radius(best_pts)
    
    np.random.seed(42)
    
    # Generate diverse initializations
    initializations = []
    initializations.append(('fibonacci', _fibonacci_points(N)))
    initializations.append(('random1', _random_points(N, seed=42)))
    initializations.append(('random2', _random_points(N, seed=123)))
    initializations.append(('random3', _random_points(N, seed=456)))
    
    # Icosahedron-based: start with icosahedron vertices, then add points to fill
    ico = _icosahedron_base()
    ico_expanded = []
    for v in ico:
        ico_expanded.append(v)
    # Add midpoints and more points to reach 50
    while len(ico_expanded) < N:
        # Add random points near existing ones
        idx = len(ico_expanded) % len(ico)
        base = ico[idx]
        noise = 0.2 * np.random.randn(3)
        new_pt = base + noise
        ico_expanded.append(_normalize(new_pt))
    initializations.append(('icosahedron', _normalize(np.array(ico_expanded[:N]))))
    
    for name, init_pts in initializations:
        if time.time() > deadline:
            break
        
        pts = init_pts.copy()
        
        # Phase 1: Lloyd relaxation for good local structure
        pts = _spherical_lloyd(pts, max_iter=80)
        
        # Phase 2: Nelder-Mead optimization on smooth surrogate
        if time.time() < deadline - 30:
            pts_nm = _optimize_with_nm(pts, max_iter=150, timeout=deadline - time.time() - 5, start_time=start_time)
            cr_nm = _covering_radius(pts_nm)
            if cr_nm < _covering_radius(pts):
                pts = pts_nm
        
        # Phase 3: High-precision facet shrink polish
        if time.time() < deadline - 10:
            pts_polished = _facet_shrink_polish(pts, max_iter=300, initial_step=0.2, decay=0.995, beta_start=10.0, beta_end=80.0)
            cr_polished = _covering_radius(pts_polished)
            if cr_polished < _covering_radius(pts):
                pts = pts_polished
        
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    # Final polish on best solution
    if time.time() < deadline - 5:
        final_polish = _facet_shrink_polish(best_pts, max_iter=500, initial_step=0.1, decay=0.998, beta_start=20.0, beta_end=100.0)
        cr_final = _covering_radius(final_polish)
        if cr_final < best_cr:
            best_pts = final_polish
            best_cr = cr_final
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
