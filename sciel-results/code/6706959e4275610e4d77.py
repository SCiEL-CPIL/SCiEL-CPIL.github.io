import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms

def _spherical_lloyd_step(points, step_size=0.1):
    """
    One Lloyd iteration: move each point toward the area-weighted centroid
    of its Voronoi cell, approximated by the circumcenters of its hull facets.
    Points are moved proportionally to how much their local circumradius
    exceeds the global median (balancing strategy).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points.copy()
    
    radii, centers, valid = _facet_circumradii(points, hull.simplices)
    if not np.any(valid):
        return points.copy()
    
    r_valid = radii[valid]
    c_valid = centers[valid]
    f_valid = hull.simplices[valid]
    
    # For each point, collect the circumcenters of all facets containing it
    # Weighted by the facet's circumradius (larger radius = more "room" to improve)
    target = np.zeros_like(points)
    weight = np.zeros(N)
    
    for k in range(len(f_valid)):
        tri = f_valid[k]
        c = c_valid[k]
        r_k = r_valid[k]
        w = max(r_k, 1e-10)  # weight by circumradius
        for idx in tri:
            target[idx] += w * c
            weight[idx] += w
    
    # Only move points that have at least one facet
    moved_mask = weight > 1e-15
    if not np.any(moved_mask):
        return points.copy()
    
    # Compute target direction (toward weighted centroid)
    new_points = points.copy()
    for idx in np.where(moved_mask)[0]:
        t = target[idx] / weight[idx]
        # Project target onto tangent plane at points[idx]
        dot_tp = np.dot(t, points[idx])
        tangent_t = t - dot_tp * points[idx]
        norm_tt = np.linalg.norm(tangent_t)
        if norm_tt > 1e-12:
            # Move toward the tangent centroid
            new_points[idx] = points[idx] + step_size * tangent_t / norm_tt
    
    new_points = _normalize(new_points)
    return new_points

def _lloyd_optimize(points, max_iters=200, base_step=0.05, min_step=1e-5):
    """
    Iterative spherical Lloyd's algorithm with adaptive step size.
    Step size decreases if covering radius increases.
    """
    best_points = points.copy()
    best_radius = _covering_radius_rad(points)
    step = base_step
    
    for it in range(max_iters):
        # Compute current covering radius
        try:
            hull = ConvexHull(points)
            radii, centers, valid = _facet_circumradii(points, hull.simplices)
            if not np.any(valid):
                break
            r_valid = radii[valid]
            max_r = np.max(r_valid)
            median_r = np.median(r_valid)
        except Exception:
            break
        
        # Adaptive step: larger when radii are imbalanced, smaller when balanced
        spread = max_r - median_r
        if spread < 1e-6:
            step *= 0.8
        elif spread > 0.1:
            step = min(step * 1.1, 0.15)
        
        # Perform Lloyd step
        new_points = _spherical_lloyd_step(points, step)
        
        # Check improvement
        new_radius = _covering_radius_rad(new_points)
        if new_radius < best_radius - 1e-8:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points
        else:
            # No improvement, reduce step
            step *= 0.7
            if step < min_step:
                break
            points = best_points.copy()
    
    return best_points, best_radius

def _facet_circumradii(points, simplices):
    """
    Compute circumradii and circumcenters for all hull facets (vectorized).
    Returns:
        radii: (M,) array of angular circumradii
        centers: (M, 3) array of circumcenters on sphere
        mask: (M,) boolean array of valid facets
    """
    M = len(simplices)
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    norm_cross = np.linalg.norm(cross, axis=1)
    
    valid = norm_cross > 1e-14
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    if np.any(valid):
        c = cross[valid] / norm_cross[valid, np.newaxis]
        
        centroid = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        dot_c_centroid = np.sum(c * centroid, axis=1)
        flip = dot_c_centroid < 0
        c[flip] = -c[flip]
        
        centers[valid] = c
        dot_val = np.clip(np.sum(c * p1[valid], axis=1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        radii[valid] = np.where(dist > np.pi / 2, np.pi - dist, dist)
    
    return radii, centers, valid

def _covering_radius_rad(points):
    """
    Compute the covering radius of a set of points on the unit sphere (in radians).
    """
    try:
        hull = ConvexHull(points)
        radii, centers, valid = _facet_circumradii(points, hull.simplices)
        if not np.any(valid):
            return np.pi
        return float(np.max(radii[valid]))
    except Exception:
        return np.pi

def _covering_radius_deg(points):
    return _covering_radius_rad(points) * 180.0 / np.pi

def _worst_facet_polish(points, max_iters=50, base_step=0.002):
    """
    Local polish: iteratively move vertices of the worst facet toward its
    circumcenter, then re-evaluate. Different from Lloyd (which moves all
    points globally) — this focuses on the single worst facet.
    """
    best_points = points.copy()
    best_radius = _covering_radius_rad(points)
    step = base_step
    
    for _ in range(max_iters):
        try:
            hull = ConvexHull(points)
            radii, centers, valid = _facet_circumradii(points, hull.simplices)
            if not np.any(valid):
                break
        except Exception:
            break
        
        r_valid = radii[valid]
        c_valid = centers[valid]
        f_valid = hull.simplices[valid]
        
        max_idx = np.argmax(r_valid)
        max_r = r_valid[max_idx]
        worst_facet = f_valid[max_idx]
        worst_center = c_valid[max_idx]
        
        new_points = points.copy()
        for idx in worst_facet:
            v = worst_center - new_points[idx]
            dot_vp = np.dot(v, new_points[idx])
            tangent_v = v - dot_vp * new_points[idx]
            norm_tv = np.linalg.norm(tangent_v)
            if norm_tv > 1e-12:
                new_points[idx] += step * tangent_v / norm_tv
        
        new_points = _normalize(new_points)
        new_radius = _covering_radius_rad(new_points)
        
        if new_radius < best_radius - 1e-9:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points
        else:
            step *= 0.7
            if step < 1e-7:
                break
            points = best_points.copy()
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    rng = np.random.default_rng(42)
    
    # Phase 0: Generate diverse initializations
    inits = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_pts = _normalize(fib_pts)
    
    # Perturbed Fibonacci
    fib_pert = fib_pts + rng.normal(0, 0.05, fib_pts.shape)
    fib_pert = _normalize(fib_pert)
    
    inits.append(fib_pts)
    inits.append(fib_pert)
    
    # 2. Random points
    for _ in range(5):
        pts = rng.normal(0, 1, (N, 3))
        pts = _normalize(pts)
        inits.append(pts)
    
    # 3. Simple electrostatic refinement (fast, vectorized)
    def electrostatic_refine(pts, n_iters=100, lr=0.01):
        pts = pts.copy()
        for _ in range(n_iters):
            D = pts @ pts.T
            chord2 = 2.0 - 2.0 * D
            np.fill_diagonal(chord2, 1.0)
            W = 1.0 / chord2
            np.fill_diagonal(W, 0.0)
            row_sum = W.sum(axis=1)
            forces = pts * row_sum[:, None] - (W @ pts)
            forces = forces - np.einsum('ij,ij->i', forces, pts)[:, None] * pts
            norms = np.linalg.norm(forces, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            forces /= norms
            pts = pts + lr * forces
            pts = _normalize(pts)
        return pts
    
    if time_left() > 15.0:
        elec_pts = electrostatic_refine(fib_pts, n_iters=150, lr=0.005)
        inits.append(elec_pts)
    
    # Evaluate initializations and pick the best
    best_points = None
    best_radius = np.pi
    
    for init in inits:
        if time_left() < 5.0:
            break
        rad = _covering_radius_rad(init)
        if rad < best_radius:
            best_radius = rad
            best_points = init.copy()
    
    if best_points is None:
        best_points = fib_pts
        best_radius = _covering_radius_rad(fib_pts)
    
    # Phase 1: Spherical Lloyd's algorithm (Voronoi-centroid relaxation)
    # This is the core structural change: instead of gradient-based optimization,
    # we iteratively move each point toward the centroid of its Voronoi cell
    # (approximated by weighted circumcenters of adjacent hull facets).
    
    if time_left() > 30.0:
        lloyd_pts, lloyd_radius = _lloyd_optimize(
            best_points, max_iters=300, base_step=0.05, min_step=1e-5
        )
        if lloyd_radius < best_radius:
            best_radius = lloyd_radius
            best_points = lloyd_pts
    
    # Phase 2: Try Lloyd from other good initializations
    for init in inits[1:4]:  # try a few more starting points
        if time_left() < 20.0:
            break
        rad = _covering_radius_rad(init)
        if rad < best_radius + 0.02:  # only if close to best
            lloyd_pts, lloyd_radius = _lloyd_optimize(
                init, max_iters=200, base_step=0.05, min_step=1e-5
            )
            if lloyd_radius < best_radius:
                best_radius = lloyd_radius
                best_points = lloyd_pts
    
    # Phase 3: Worst-facet local polish for fine-tuning
    if time_left() > 15.0:
        polished_pts, polished_radius = _worst_facet_polish(
            best_points, max_iters=80, base_step=0.003
        )
        if polished_radius < best_radius:
            best_radius = polished_radius
            best_points = polished_pts
    
    # Phase 4: Final Lloyd pass with smaller steps for convergence
    if time_left() > 10.0:
        lloyd_pts, lloyd_radius = _lloyd_optimize(
            best_points, max_iters=100, base_step=0.02, min_step=1e-6
        )
        if lloyd_radius < best_radius:
            best_radius = lloyd_radius
            best_points = lloyd_pts
    
    # Phase 5: Final worst-facet polish
    if time_left() > 5.0:
        polished_pts, polished_radius = _worst_facet_polish(
            best_points, max_iters=50, base_step=0.001
        )
        if polished_radius < best_radius:
            best_radius = polished_radius
            best_points = polished_pts
    
    return best_points
