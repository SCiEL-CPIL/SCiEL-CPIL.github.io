"""Sphere covering with 130 equal caps - combined approach with repulsion refinement."""
import numpy as np
import time
from scipy.spatial import ConvexHull
import warnings

N = 130

# EVOLVE-BLOCK-START

def _normalize(pts):
    """Normalize points to unit sphere."""
    pts = np.array(pts, dtype=float)
    norms = np.linalg.norm(pts, axis=1)
    norms[norms < 1e-14] = 1e-14
    return pts / norms[:, None]

def _covering_radius(points):
    """Compute covering radius in degrees via convex hull facets."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
        faces = hull.simplices
        a = points[faces[:, 0]]
        b = points[faces[:, 1]]
        c = points[faces[:, 2]]
        
        # Compute circumradius for each triangular face
        # For a triangle on the unit sphere with vertices a, b, c:
        # The circumcenter is proportional to a + b + c (if not degenerate)
        # The circumradius r satisfies cos(r) = (a+b+c)·a / |a+b+c|
        # Actually, for unit sphere, the circumcenter n = (a+b+c)/|a+b+c|
        # and cos(r) = n·a
        
        s = a + b + c
        s_norm = np.linalg.norm(s, axis=1)
        
        radii = np.zeros(len(faces))
        valid = s_norm > 1e-10
        
        if np.any(valid):
            n = s[valid] / s_norm[valid, np.newaxis]
            cos_r = np.clip(n @ a[valid], -1.0, 1.0)
            r = np.arccos(cos_r)
            # The circumradius should be <= pi/2 for proper facets
            # If r > pi/2, the facet is "large" and we take pi - r
            r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
            radii[valid] = r
        
        if np.any(~valid):
            # Degenerate face - use max pairwise distance
            a_deg = a[~valid]
            b_deg = b[~valid]
            c_deg = c[~valid]
            d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
            d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
            d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
            radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
        
        return float(np.degrees(np.max(radii)))
    except Exception:
        # Fallback: dense grid-based estimate
        return _covering_radius_brute(points)

def _covering_radius_brute(points, n_theta=360, n_phi=720):
    """Brute force covering radius computation on a grid."""
    points = _normalize(points)
    
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    max_dist = 0.0
    chunk_size = 20000
    for start in range(0, len(grid), chunk_size):
        end = min(start + chunk_size, len(grid))
        g = grid[start:end]
        dots = g @ points.T
        dots = np.clip(dots, -1, 1)
        max_dots = np.max(dots, axis=1)
        min_angs = np.arccos(max_dots)
        max_dist = max(max_dist, np.max(min_angs))
    
    return float(np.degrees(max_dist))

def _fibonacci_spiral(n):
    """Generate points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(points)

def _lloyd_iteration(points, n_iter=50, tol=1e-10):
    """Lloyd's algorithm for spherical k-means / Voronoi centroid refinement."""
    points = _normalize(points)
    n = len(points)
    
    for it in range(n_iter):
        n_theta = 360
        n_phi = 720
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        x = np.sin(TH) * np.cos(PH)
        y = np.sin(TH) * np.sin(PH)
        z = np.cos(TH)
        grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        dots = grid @ points.T
        nearest = np.argmax(dots, axis=1)
        
        new_points = np.zeros_like(points)
        counts = np.zeros(n)
        for i in range(n):
            mask = nearest == i
            if np.sum(mask) > 0:
                new_points[i] = grid[mask].mean(axis=0)
                counts[i] = np.sum(mask)
        
        for i in range(n):
            if counts[i] == 0:
                new_points[i] = points[i]
        
        new_points = _normalize(new_points)
        
        diff = np.linalg.norm(new_points - points)
        points = new_points
        if diff < tol:
            break
    
    return points

def _repulsion_refine(pts, max_iter=200, alpha_init=0.008, decay=0.995):
    """
    Refine using repulsion energy gradient to spread points more evenly.
    This maximizes the minimum pairwise distance, which correlates with minimizing covering radius.
    """
    x = _normalize(pts)
    alpha = alpha_init
    
    for it in range(max_iter):
        dot = x @ x.T
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        
        # Use a steeper potential to push apart close points
        w = 1.0 / (dist_sq ** 3)
        np.fill_diagonal(w, 0.0)
        
        w_grad = 3.0 * (dist_sq ** (-2.5))
        np.fill_diagonal(w_grad, 0.0)
        
        grad = w_grad @ x
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        x = x + alpha * grad / grad_norm
        x = _normalize(x)
        
        alpha *= decay
    
    return x

def _facet_polish(points, max_time=30.0):
    """Polish by iteratively moving worst-facet vertices toward their circumcenter."""
    points = _normalize(points)
    start = time.time()
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    lr = 0.05
    no_improve = 0
    
    while time.time() - start < max_time and no_improve < 50:
        points = _normalize(points)
        
        try:
            hull = ConvexHull(points)
            faces = hull.simplices
        except Exception:
            break
        
        max_circ = 0.0
        worst_face = None
        for face in faces:
            p1, p2, p3 = points[face[0]], points[face[1]], points[face[2]]
            n = np.cross(p2 - p1, p3 - p1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            r1 = np.arccos(np.clip(n @ p1, -1, 1))
            r2 = np.arccos(np.clip(-n @ p1, -1, 1))
            r = min(r1, r2)
            if r > max_circ:
                max_circ = r
                worst_face = face
        
        if worst_face is None:
            break
        
        p1, p2, p3 = points[worst_face[0]], points[worst_face[1]], points[worst_face[2]]
        n = np.cross(p2 - p1, p3 - p1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            break
        n = n / n_norm
        c = n if (n @ p1) > 0 else -n
        
        for idx in worst_face:
            p = points[idx]
            d = c - p
            d = d - (d @ p) * p
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-12:
                d = d / d_norm
                points[idx] = points[idx] + lr * d
        
        points = _normalize(points)
        
        new_radius = _covering_radius(points)
        if new_radius < best_radius - 1e-8:
            best_radius = new_radius
            best_points = points.copy()
            no_improve = 0
        else:
            no_improve += 1
            points = best_points.copy()
            lr *= 0.85
    
    return best_points

def _soft_max_loss(x_flat, grid_pts, temperature=30.0):
    """
    Compute a smooth approximation of the covering radius using soft-max over
    soft-min distances. Returns (value, gradient) with respect to x_flat.
    """
    pts = x_flat.reshape(N, 3)
    pts = _normalize(pts)
    
    M = len(grid_pts)
    
    # Angular distances: M x N
    dot = grid_pts @ pts.T
    dot = np.clip(dot, -1.0, 1.0)
    ang_dists = np.arccos(dot)
    
    # Soft-min over points for each grid point
    T_min = 30.0
    exp_neg = np.exp(-T_min * ang_dists)
    sum_exp = np.sum(exp_neg, axis=1, keepdims=True)
    sum_exp[sum_exp < 1e-300] = 1e-300
    soft_min = -1.0 / T_min * np.log(sum_exp)[:, 0]
    
    # Weights for soft-min
    w = exp_neg / sum_exp  # M x N
    
    # Soft-max over grid points
    T_max = 30.0
    max_sm = np.max(soft_min)
    shifted = soft_min - max_sm
    exp_shifted = np.exp(T_max * shifted)
    sum_exp_max = np.sum(exp_shifted)
    if sum_exp_max < 1e-300:
        sum_exp_max = 1e-300
    val = max_sm + (1.0 / T_max) * np.log(sum_exp_max)
    
    # Gradient of soft-max w.r.t soft_min
    w_max = exp_shifted / sum_exp_max  # M
    
    # Compute gradient efficiently
    sqrt_term = np.sqrt(np.clip(1.0 - dot * dot, 1e-10, 1.0))  # M x N
    G = w_max[:, np.newaxis] * w / sqrt_term  # M x N
    grad_pts = -(G.T @ grid_pts)  # N x 3
    
    grad_flat = grad_pts.reshape(-1)
    
    return float(val), grad_flat

def _gradient_descent_smooth(pts, grid_pts, max_iter=200, alpha_init=0.01, decay=0.98, temperature=30.0):
    """
    Gradient descent on the smooth soft-max loss to minimize covering radius.
    """
    x = pts.reshape(-1).copy()
    best_x = x.copy()
    best_val = _covering_radius(pts)
    
    alpha = alpha_init
    for it in range(max_iter):
        val, grad = _soft_max_loss(x, grid_pts, temperature)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        x = x - alpha * grad
        
        p = x.reshape(N, 3)
        p = _normalize(p)
        
        exact_r = _covering_radius(p)
        if exact_r < best_val:
            best_val = exact_r
            best_x = x.copy()
        
        alpha *= decay
        temperature = max(10.0, temperature * 0.995)
    
    p = best_x.reshape(N, 3)
    return _normalize(p)

def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    # Stage 1: Fibonacci spiral initialization
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Stage 2: Lloyd's algorithm for global balance
    if time.time() < deadline - 80:
        lloyd_points = _lloyd_iteration(points, n_iter=60)
        lloyd_radius = _covering_radius(lloyd_points)
        if lloyd_radius < best_radius:
            best_points = lloyd_points
            best_radius = lloyd_radius
    
    # Stage 3: Repulsion-based refinement (stronger, more iterations)
    if time.time() < deadline - 40:
        rep_points = _repulsion_refine(best_points, max_iter=150, alpha_init=0.006, decay=0.993)
        rep_radius = _covering_radius(rep_points)
        if rep_radius < best_radius:
            best_points = rep_points
            best_radius = rep_radius
    
    # Stage 4: Smooth gradient descent on soft-max loss
    if time.time() < deadline - 25:
        n_theta = 200
        n_phi = 400
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        gx = np.sin(TH) * np.cos(PH)
        gy = np.sin(TH) * np.sin(PH)
        gz = np.cos(TH)
        grid = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        
        smooth_points = _gradient_descent_smooth(best_points, grid, max_iter=100, alpha_init=0.005, decay=0.99)
        smooth_radius = _covering_radius(smooth_points)
        if smooth_radius < best_radius:
            best_points = smooth_points
            best_radius = smooth_radius
    
    # Stage 5: Facet-based polishing
    remaining = deadline - time.time()
    if remaining > 10:
        polished = _facet_polish(best_points, max_time=min(20, remaining - 5))
        polish_radius = _covering_radius(polished)
        if polish_radius < best_radius:
            best_points = polished
            best_radius = polish_radius
    
    # Final normalization
    best_points = _normalize(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
