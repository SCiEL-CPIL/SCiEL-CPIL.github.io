"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math

N = 50
_SEED = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius (in radians) using convex hull facets.
    For each triangular facet, the circumcenter is the normalized normal
    to the facet plane. The angular radius is arccos(|n·v|).
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: brute force on a fine grid
        n_theta = 40
        n_phi = 80
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        T, P = np.meshgrid(theta, phi, indexing='ij')
        Y = np.column_stack([
            np.sin(T) * np.cos(P).ravel(),
            np.sin(T) * np.sin(P).ravel(),
            np.cos(T).ravel()
        ])
        Y = _normalize(Y)
        dots = Y @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        return np.max(min_dists)
    
    max_rad = 0.0
    for simplex in hull.simplices:
        v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        n = np.cross(v1 - v2, v1 - v3)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            continue
        n = n / norm_n
        dot_val = abs(n @ v1)
        dot_val = min(dot_val, 1.0)
        rad = np.arccos(dot_val)
        if rad > max_rad:
            max_rad = rad
    
    return max_rad

def _covering_radius_degrees(pts):
    return np.degrees(_covering_radius(pts))

def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _spherical_lloyd(pts, max_iter=200, tol=1e-8):
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

def _worst_facet_circumcenter_projection(pts, max_iter=500, decay=0.998, initial_step=0.5):
    """
    Deterministic iterative method: identifies the convex hull facet with the
    largest spherical circumradius and moves its three vertices along the great
    circle toward the facet's circumcenter. This directly minimizes the maximum
    empty cap without stochastic noise or gradient-based energy landscapes.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    step_size = initial_step
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Find the facet with the largest circumradius
        max_facet_rad = 0.0
        max_facet = None
        max_cc = None
        
        for simplex in hull.simplices:
            v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            n = np.cross(v1 - v2, v1 - v3)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                continue
            n = n / norm_n
            
            # Choose sign so normal points outward (away from origin)
            if n @ v1 < 0:
                n = -n
            
            dot_val = abs(n @ v1)
            dot_val = min(dot_val, 1.0)
            rad = np.arccos(dot_val)
            
            if rad > max_facet_rad:
                max_facet_rad = rad
                max_facet = simplex
                max_cc = n
        
        if max_facet is None or max_facet_rad < 1e-8:
            break
        
        # Move each vertex of the worst facet toward the circumcenter
        cc = max_cc
        for idx in max_facet:
            p = pts[idx]
            dot = p @ cc
            dot = np.clip(dot, -1.0, 1.0)
            angle = np.arccos(dot)
            if angle < 1e-10:
                continue
            
            # Slerp by a fraction of the angle
            new_angle = angle * (1.0 - step_size)
            sin_a = np.sin(angle)
            if sin_a < 1e-10:
                continue
            sin_na = np.sin(new_angle)
            cos_na = np.cos(new_angle)
            coeff_p = sin_na / sin_a
            coeff_cc = (cos_na - dot * sin_na / sin_a)
            new_p = coeff_p * p + coeff_cc * cc
            new_p = new_p / np.linalg.norm(new_p)
            pts[idx] = new_p
        
        pts = _normalize(pts)
        step_size *= decay
        if step_size < 1e-8:
            break
    
    return pts

def _optimize_facet_shrink(pts, max_iter=300, initial_step=0.5, decay=0.995, beta_start=2.0, beta_end=20.0):
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
            
            n_face = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n_face)
            if norm_n < 1e-12:
                continue
            n_face = n_face / norm_n
            if np.dot(n_face, a) < 0:
                n_face = -n_face
                
            dot_val = abs(n_face @ a)
            dot_val = min(dot_val, 1.0)
            cr = np.arccos(dot_val)
            
            max_cr_global = max(max_cr_global, cr)
            facet_data.append((simplex, n_face, cr))
            
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
        step_size *= decay
        
        if step_size < 1e-8:
            break

    return pts

def _soft_covering_radius(pts, beta=50.0):
    """
    Smooth approximation of the covering radius using log-sum-exp over a grid.
    """
    pts = _normalize(pts)
    n_theta = 25
    n_phi = 50
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    Y = np.column_stack([
        np.sin(T) * np.cos(P).ravel(),
        np.sin(T) * np.sin(P).ravel(),
        np.cos(T).ravel()
    ])
    Y = _normalize(Y)
    dots = Y @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    max_d = np.max(min_dists)
    soft_max = max_d + np.log(np.sum(np.exp(beta * (min_dists - max_d)))) / beta
    return soft_max

def _repulsion_energy(pts, power=20):
    pts = _normalize(pts)
    dots = pts @ pts.T
    np.fill_diagonal(dots, 0)
    dots = np.clip(dots, -1.0, 0.999999)
    dists = np.arccos(dots)
    dists = np.maximum(dists, 1e-10)
    energy = np.sum(1.0 / (dists ** power))
    return energy

def _optimize_with_beta_cascade(pts, betas=[15, 50, 150, 300, 500], max_iter_per_stage=2000):
    """
    Multi-stage beta cascade optimization.
    Start with low beta (broad landscape) and increase to high beta (precise).
    """
    pts = _normalize(pts)
    for beta in betas:
        def obj(x):
            p = _normalize(x.reshape(N, 3))
            return _soft_covering_radius(p, beta=beta)
        
        try:
            res = minimize(
                obj,
                pts.ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': max_iter_per_stage,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            pts = _normalize(res.x.reshape(N, 3))
        except Exception:
            pass
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    # Initial point: Fibonacci spiral
    best_pts = _fibonacci_spiral()
    best_radius = _covering_radius(best_pts)
    
    np.random.seed(_SEED)
    
    # Phase 1: Repulsion-based optimization to spread points
    for restart in range(3):
        if time.time() > deadline - 60.0:
            break
        
        if restart == 0:
            x0 = best_pts.copy()
        else:
            x0 = np.random.randn(N, 3)
            x0 = _normalize(x0)
        
        def obj_repulsion(x):
            p = _normalize(x.reshape(N, 3))
            return -_repulsion_energy(p, power=10)
        
        try:
            res = minimize(
                obj_repulsion,
                x0.ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            x_opt = _normalize(res.x.reshape(N, 3))
            rad = _covering_radius(x_opt)
            if rad < best_radius:
                best_radius = rad
                best_pts = x_opt.copy()
        except Exception:
            continue
    
    # Phase 2: Spherical Lloyd's algorithm (Voronoi relaxation)
    if time.time() < deadline - 50.0:
        polished = _spherical_lloyd(best_pts, max_iter=200)
        rad = _covering_radius(polished)
        if rad < best_radius:
            best_radius = rad
            best_pts = polished.copy()
    
    # Phase 3: Worst-facet circumcenter projection (deterministic geometric shrinkage)
    if time.time() < deadline - 40.0:
        polished = _worst_facet_circumcenter_projection(best_pts, max_iter=500)
        rad = _covering_radius(polished)
        if rad < best_radius:
            best_radius = rad
            best_pts = polished.copy()
    
    # Phase 4: Facet shrink with weighted circumcenters
    if time.time() < deadline - 30.0:
        polished = _optimize_facet_shrink(best_pts, max_iter=300)
        rad = _covering_radius(polished)
        if rad < best_radius:
            best_radius = rad
            best_pts = polished.copy()
    
    # Phase 5: Beta cascade optimization
    if time.time() < deadline - 20.0:
        polished = _optimize_with_beta_cascade(best_pts, betas=[15, 50, 150], max_iter_per_stage=1500)
        rad = _covering_radius(polished)
        if rad < best_radius:
            best_radius = rad
            best_pts = polished.copy()
    
    # Phase 6: Final worst-facet circumcenter projection
    if time.time() < deadline - 10.0:
        polished = _worst_facet_circumcenter_projection(best_pts, max_iter=300)
        rad = _covering_radius(polished)
        if rad < best_radius:
            best_radius = rad
            best_pts = polished.copy()
    
    # Phase 7: Final high-beta polish
    if time.time() < deadline - 5.0:
        polished = _optimize_with_beta_cascade(best_pts, betas=[300, 500], max_iter_per_stage=1000)
        rad = _covering_radius(polished)
        if rad < best_radius:
            best_radius = rad
            best_pts = polished.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
