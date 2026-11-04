import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])
N_TRIPLES = TRIPLE_INDICES.shape[0]

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])

    min_area = 0.5 * np.min(np.abs(cross))

    return min_area / hull_area

def _triangle_areas(points):
    """Compute all triangle areas (not doubled) and hull area."""
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        hull_area = 1.0
    return areas, hull_area

def _soft_min_score(points, beta=50.0):
    """
    Smooth soft-min of triangle areas / hull_area.
    As beta -> inf, this approaches the true min-area / hull-area.
    """
    areas, hull_area = _triangle_areas(points)
    if hull_area <= 1e-9:
        return -1e6
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    # Numerically stable version
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = min_a - np.log(np.sum(np.exp(-beta * (areas - min_a)))) / beta
    # Actually: soft_min = -1/beta * log(sum(exp(-beta * a_i)))
    # = min_a - (1/beta)*log(sum(exp(-beta*(a_i - min_a))))
    soft_min_val = min_a - (1.0/beta) * np.log(np.sum(np.exp(-beta * shifted)))
    return -soft_min_val / hull_area

def _soft_min_score_polar(polar_flat, beta=50.0):
    """Soft-min objective in polar coordinates."""
    polar = polar_flat.reshape(N, 2)
    r = np.maximum(polar[:, 0], 0.05)
    theta = polar[:, 1]
    points = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    return _soft_min_score(points, beta)

def _hard_min_score_flat(points_flat):
    """Hard min objective (negative score for minimization)."""
    points = points_flat.reshape(N, 2)
    return -_calculate_score(points)

def _generate_regular_polygon(r=1.0):
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return np.column_stack((x, y))

def _generate_k_ring(radii, counts, offsets):
    all_points = []
    for r, n, theta0 in zip(radii, counts, offsets):
        if n == 0:
            continue
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + theta0
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        all_points.append(np.column_stack((x, y)))
    if not all_points:
        return _generate_regular_polygon(1.0)
    return np.vstack(all_points)

def _generate_random_points(seed):
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.random(N))
    theta = rng.random(N) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def _polar_to_cartesian(polar):
    r = np.maximum(polar[:, 0], 0.01)
    theta = polar[:, 1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def _cartesian_to_polar(points):
    r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    theta = np.arctan2(points[:, 1], points[:, 0])
    return np.column_stack((r, theta))

def _polish_soft_min(points, time_limit=8.0, beta_start=20.0, beta_end=200.0):
    """
    Multi-stage soft-min polish: start with low beta (smooth) and increase
    to high beta (sharp) for convergence to true min.
    """
    start_time = time.time()
    best_score = _calculate_score(points)
    best_points = points.copy()
    
    # Stage 1: Low beta (smooth landscape, global-ish)
    n_stages = 3
    betas = np.linspace(beta_start, beta_end, n_stages)
    
    for beta in betas:
        if time.time() - start_time > time_limit - 1.0:
            break
        
        def objective(x):
            pts = x.reshape(N, 2)
            return _soft_min_score(pts, beta)
        
        try:
            res = minimize(
                objective,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            new_points = res.x.reshape(N, 2)
            new_score = _calculate_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
        except Exception:
            pass
    
    # Final hard-min polish
    if time.time() - start_time < time_limit:
        try:
            res = minimize(
                _hard_min_score_flat,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            new_points = res.x.reshape(N, 2)
            new_score = _calculate_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
        except Exception:
            pass
    
    return best_score, best_points

def _polish_polar_soft_min(polar_points, time_limit=8.0, beta_start=20.0, beta_end=200.0):
    """
    Multi-stage soft-min polish in polar coordinates.
    """
    start_time = time.time()
    best_score = _calculate_score(_polar_to_cartesian(polar_points))
    best_polar = polar_points.copy()
    
    n_stages = 3
    betas = np.linspace(beta_start, beta_end, n_stages)
    
    for beta in betas:
        if time.time() - start_time > time_limit - 1.0:
            break
        
        def objective(x):
            return _soft_min_score_polar(x, beta)
        
        try:
            res = minimize(
                objective,
                best_polar.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            new_polar = res.x.reshape(N, 2)
            new_points = _polar_to_cartesian(new_polar)
            new_score = _calculate_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_polar = new_polar
        except Exception:
            pass
    
    # Final hard-min polish in cartesian
    if time.time() - start_time < time_limit:
        best_pts = _polar_to_cartesian(best_polar)
        score_c, pts_c = _polish_soft_min(best_pts, time_limit=2.0, beta_start=beta_end, beta_end=beta_end)
        if score_c > best_score:
            best_score = score_c
            best_polar = _cartesian_to_polar(pts_c)
    
    return best_score, best_polar

def _coordinate_descent_polish(points, time_limit=5.0):
    """
    Coordinate-level coordinate descent: perturb each point individually
    using small local searches.
    """
    start_time = time.time()
    best_points = points.copy()
    best_score = _calculate_score(best_points)
    
    for _round in range(5):
        if time.time() - start_time > time_limit - 0.5:
            break
        for i in range(N):
            if time.time() - start_time > time_limit - 0.3:
                break
            # Small local search around point i
            best_local = best_points.copy()
            best_local_score = best_score
            rng = np.random.default_rng(i * 1000 + _round)
            
            for _trial in range(8):
                if time.time() - start_time > time_limit - 0.2:
                    break
                candidate = best_local.copy()
                candidate[i] += rng.normal(0, 0.05, 2)
                # Clamp to reasonable range
                candidate[i, 0] = np.clip(candidate[i, 0], -2, 2)
                candidate[i, 1] = np.clip(candidate[i, 1], -2, 2)
                s = _calculate_score(candidate)
                if s > best_local_score:
                    best_local_score = s
                    best_local = candidate
            
            if best_local_score > best_score:
                best_score = best_local_score
                best_points = best_local
    
    return best_score, best_points

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    best_score = 0.0
    best_points = None

    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Phase 1: Regular polygon + soft-min polish (cartesian)
    if time.time() < deadline - 40:
        reg = _generate_regular_polygon(1.0)
        score, pts = _polish_soft_min(reg, time_limit=10.0)
        try_update(score, pts)
        # Also try polar soft-min polish
        polar_reg = _cartesian_to_polar(reg)
        score_p, polar_p = _polish_polar_soft_min(polar_reg, time_limit=8.0)
        pts_p = _polar_to_cartesian(polar_p)
        try_update(score_p, pts_p)

    # Phase 2: Multi-ring configurations with soft-min polish
    ring_configs = [
        ([1.0], [14], [0.0]),
        ([1.0, 0.6], [8, 6], [0.0, np.pi/8]),
        ([1.0, 0.6], [7, 7], [0.0, np.pi/7]),
        ([1.0, 0.5], [9, 5], [0.0, np.pi/9]),
        ([1.0, 0.7], [10, 4], [0.0, np.pi/10]),
        ([1.0, 0.8, 0.4], [6, 4, 4], [0.0, np.pi/6, np.pi/4]),
        ([1.0, 0.5, 0.2], [7, 4, 3], [0.0, np.pi/7, np.pi/3]),
        ([1.0, 0.65], [12, 2], [0.0, np.pi/6]),
        ([1.0, 0.45], [11, 3], [0.0, np.pi/11]),
        ([1.0, 0.75], [9, 5], [0.0, np.pi/9]),
    ]

    for radii, counts, offsets in ring_configs:
        if time.time() > deadline - 30:
            break
        start_pts = _generate_k_ring(radii, counts, offsets)
        score, pts = _polish_soft_min(start_pts, time_limit=6.0)
        try_update(score, pts)

    # Phase 3: Soft-min global search in polar coords via Nelder-Mead
    if time.time() < deadline - 35:
        try:
            # Start from regular polygon in polar
            reg_polar = _cartesian_to_polar(_generate_regular_polygon(1.0))
            
            def nm_objective(x):
                if time.time() > deadline - 5:
                    return 1e6
                return _soft_min_score_polar(x, beta=30.0)
            
            res = minimize(
                nm_objective,
                reg_polar.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 500, 'maxfev': 3000, 'disp': False}
            )
            if time.time() < deadline - 20:
                de_polar = res.x.reshape(N, 2)
                de_points = _polar_to_cartesian(de_polar)
                score, pts = _polish_soft_min(de_points, time_limit=8.0)
                try_update(score, pts)
        except Exception:
            pass

    # Phase 4: Multiple random starts with soft-min polish
    for seed in range(1, 12):
        if time.time() > deadline - 20:
            break
        start_pts = _generate_random_points(seed)
        score, pts = _polish_soft_min(start_pts, time_limit=5.0)
        try_update(score, pts)

    # Phase 5: Perturbed regular polygon
    for noise_level in [0.05, 0.1, 0.2, 0.35]:
        if time.time() > deadline - 15:
            break
        rng = np.random.default_rng(100 + int(noise_level * 100))
        angles = np.linspace(0, 2*np.pi, N, endpoint=False) + rng.normal(0, noise_level, N)
        r = 1.0 + rng.normal(0, noise_level/5, N)
        r = np.maximum(r, 0.1)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        start_pts = np.column_stack((x, y))
        score, pts = _polish_soft_min(start_pts, time_limit=4.0)
        try_update(score, pts)

    # Phase 6: Coordinate descent polish on best
    if best_points is not None and time.time() < deadline - 5:
        score_cd, pts_cd = _coordinate_descent_polish(best_points, time_limit=5.0)
        try_update(score_cd, pts_cd)

    # Final: ensure best_points is not None
    if best_points is None:
        best_points = _generate_regular_polygon(1.0)

    return best_points
