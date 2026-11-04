import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 14

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area."""
    n = len(points)
    if n < 3:
        return 0.0
    
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i = i[mask].flatten()
    j = j[mask].flatten()
    k = k[mask].flatten()
    
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _all_triangle_areas(points):
    """Return all triangle areas as a flat array."""
    n = len(points)
    if n < 3:
        return np.array([0.0])
    
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i = i[mask].flatten()
    j = j[mask].flatten()
    k = k[mask].flatten()
    
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _soft_min_triangle_area(points, beta=30.0):
    """Smooth approximation of min triangle area using log-sum-exp."""
    areas = _all_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    n_tris = len(areas)
    # soft_min = -beta * log(mean(exp(-areas/beta)))
    # = -beta * (log(sum(exp(-areas/beta))) - log(n_tris))
    # Numerically stable:
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-shifted / beta)))
    soft_min = min_a - beta * (log_sum - np.log(n_tris))
    return soft_min

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _soft_score(points, beta=30.0):
    """Smooth surrogate score using soft-min triangle area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _soft_min_triangle_area(points, beta)
    return min_area / hull_area

def _build_ring_config(radii, angles_per_ring, phase_offsets):
    """Build point set from concentric rings."""
    pts = []
    for r, n_ang, phase in zip(radii, angles_per_ring, phase_offsets):
        angles = np.linspace(0, 2 * np.pi, n_ang, endpoint=False) + phase
        ring_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        pts.append(ring_pts)
    return np.vstack(pts)

def _coordinate_descent_polish(points, time_limit=15.0, max_rounds=20):
    """
    Per-point coordinate descent polish.
    For each point, try small perturbations in x and y, accepting if score improves.
    This escapes ring-parameterization constraints.
    """
    start_time = time.time()
    best_pts = points.copy()
    best_score = _score(best_pts)
    
    # Precompute triangle indices
    n = len(best_pts)
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    tri_i = i_idx[mask].flatten()
    tri_j = j_idx[mask].flatten()
    tri_k = k_idx[mask].flatten()
    
    # For each point, precompute which triangles involve it
    point_tri_map = []
    for p in range(n):
        tri_mask = (tri_i == p) | (tri_j == p) | (tri_k == p)
        point_tri_map.append(np.where(tri_mask)[0])
    
    # Adaptive step sizes
    base_step = 0.05
    shrink = 0.7
    expand = 1.3
    
    for rnd in range(max_rounds):
        if time.time() > start_time + time_limit:
            break
        
        improved = False
        for p in range(n):
            if time.time() > start_time + time_limit:
                break
            
            step = base_step * (shrink ** rnd)
            for axis in range(2):
                for direction in [-1, 1]:
                    trial = best_pts.copy()
                    trial[p, axis] += direction * step
                    s = _score(trial)
                    if s > best_score + 1e-12:
                        best_pts = trial
                        best_score = s
                        improved = True
        
        if not improved:
            break
    
    return best_score, best_pts

def _coordinate_descent_polish_soft(points, time_limit=15.0, max_rounds=30, beta=20.0):
    """
    Per-point coordinate descent polish using soft score for smoother landscape.
    """
    start_time = time.time()
    best_pts = points.copy()
    best_score = _score(best_pts)
    best_soft = _soft_score(best_pts, beta)
    
    base_step = 0.08
    shrink = 0.75
    
    for rnd in range(max_rounds):
        if time.time() > start_time + time_limit:
            break
        
        improved = False
        step = base_step * (shrink ** rnd)
        for p in range(len(best_pts)):
            if time.time() > start_time + time_limit:
                break
            for axis in range(2):
                for direction in [-1, 1]:
                    trial = best_pts.copy()
                    trial[p, axis] += direction * step
                    s_soft = _soft_score(trial, beta)
                    if s_soft > best_soft + 1e-10:
                        best_pts = trial
                        best_soft = s_soft
                        s_exact = _score(best_pts)
                        if s_exact > best_score:
                            best_score = s_exact
                        improved = True
        
        if not improved:
            break
    
    return best_score, best_pts

def _optimize_ring_config(n1, n2, n3, r_init, phase_init, time_limit=8.0):
    """Optimize a 3-ring configuration."""
    start_time = time.time()
    
    def build_from_params(x):
        r1, r2, r3, p1, p2, p3 = x
        r1 = max(0.3, r1)
        r2 = max(0.05, min(r1 * 0.95, r2))
        r3 = max(0.01, min(r2 * 0.9, r3))
        p1 = p1 % (2 * np.pi)
        p2 = p2 % (2 * np.pi)
        p3 = p3 % (2 * np.pi)
        return _build_ring_config([r1, r2, r3], [n1, n2, n3], [p1, p2, p3])
    
    def neg_soft_score_params(x, beta=25.0):
        pts = build_from_params(x)
        return -_soft_score(pts, beta)
    
    x0 = np.array([r_init[0], r_init[1], r_init[2], phase_init[0], phase_init[1], phase_init[2]])
    best_score = _score(build_from_params(x0))
    best_x = x0.copy()
    
    bounds = [(0.3, 2.5), (0.05, 2.5), (0.01, 2.5), (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
    
    # Multiple restarts with smooth objective
    rng = np.random.RandomState(42 + n1*100 + n2*10 + n3)
    starts = [x0]
    for _ in range(8):
        x_pert = x0 + rng.normal(0, 0.2, size=6)
        x_pert[0] = max(0.3, x_pert[0])
        x_pert[1] = max(0.05, min(x_pert[0]*0.95, x_pert[1]))
        x_pert[2] = max(0.01, min(x_pert[1]*0.9, x_pert[2]))
        x_pert[3:] = x_pert[3:] % (2*np.pi)
        starts.append(x_pert)
    
    for xi in starts:
        if time.time() > start_time + time_limit * 0.5:
            break
        try:
            res = minimize(
                lambda x: neg_soft_score_params(x, beta=25.0), xi, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-9, 'adaptive': True}
            )
            s_exact = _score(build_from_params(res.x))
            if s_exact > best_score:
                best_score = s_exact
                best_x = res.x.copy()
        except Exception:
            pass
    
    best_pts = build_from_params(best_x)
    return best_score, best_pts

def _optimize_two_ring_config(n1, n2, r_init, phase_init, time_limit=8.0):
    """Optimize a 2-ring configuration."""
    start_time = time.time()
    
    def build_from_params(x):
        r1, r2, p1, p2 = x
        r1 = max(0.3, r1)
        r2 = max(0.05, min(r1 * 0.95, r2))
        p1 = p1 % (2 * np.pi)
        p2 = p2 % (2 * np.pi)
        return _build_ring_config([r1, r2], [n1, n2], [p1, p2])
    
    def neg_soft_score_params(x, beta=25.0):
        pts = build_from_params(x)
        return -_soft_score(pts, beta)
    
    x0 = np.array([r_init[0], r_init[1], phase_init[0], phase_init[1]])
    best_score = _score(build_from_params(x0))
    best_x = x0.copy()
    
    bounds = [(0.3, 2.5), (0.05, 2.5), (0, 2*np.pi), (0, 2*np.pi)]
    
    rng = np.random.RandomState(42 + n1*100 + n2)
    starts = [x0]
    for _ in range(8):
        x_pert = x0 + rng.normal(0, 0.2, size=4)
        x_pert[0] = max(0.3, x_pert[0])
        x_pert[1] = max(0.05, min(x_pert[0]*0.95, x_pert[1]))
        x_pert[2:] = x_pert[2:] % (2*np.pi)
        starts.append(x_pert)
    
    for xi in starts:
        if time.time() > start_time + time_limit * 0.5:
            break
        try:
            res = minimize(
                lambda x: neg_soft_score_params(x, beta=25.0), xi, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-9, 'adaptive': True}
            )
            s_exact = _score(build_from_params(res.x))
            if s_exact > best_score:
                best_score = s_exact
                best_x = res.x.copy()
        except Exception:
            pass
    
    best_pts = build_from_params(best_x)
    return best_score, best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Strategy: 
    # Phase 1: Explore ring configurations with smooth objective (fast)
    # Phase 2: Coordinate descent polish on best candidates
    # Phase 3: Final exact refinement
    
    # Ring configurations
    ring_configs_3 = [
        (6, 5, 3), (5, 5, 4), (7, 4, 3), (5, 4, 5),
        (4, 5, 5), (6, 4, 4), (4, 6, 4), (7, 5, 2),
        (2, 6, 6), (5, 6, 3), (3, 6, 5), (4, 4, 6),
    ]
    
    ring_configs_2 = [
        (7, 7), (6, 8), (8, 6), (5, 9), (9, 5), (4, 10),
    ]
    
    # Phase 1: Ring optimization with smooth objective
    candidates = []  # list of (score, points)
    
    for (n1, n2, n3) in ring_configs_3:
        if time.time() > deadline - 50:
            break
        for r_ratio in [0.35, 0.5, 0.65, 0.8]:
            if time.time() > deadline - 50:
                break
            r_init = [1.0, 0.3 + 0.5*r_ratio, 0.1 + 0.35*r_ratio]
            r_init[1] = min(r_init[1], r_init[0] * 0.9)
            r_init[2] = min(r_init[2], r_init[1] * 0.85)
            for phase_shift in [0.0, np.pi/(2*n1), np.pi/n1]:
                if time.time() > deadline - 50:
                    break
                phase_init = [0.0, phase_shift, 2*phase_shift]
                s, p = _optimize_ring_config(n1, n2, n3, r_init, phase_init, time_limit=3.0)
                if s > best_overall_score:
                    best_overall_score = s
                    best_overall_points = p
                candidates.append((s, p))
    
    for (n1, n2) in ring_configs_2:
        if time.time() > deadline - 35:
            break
        for r_ratio in [0.3, 0.5, 0.7, 0.85]:
            if time.time() > deadline - 35:
                break
            r_init = [1.0, 0.2 + 0.7*r_ratio]
            r_init[1] = min(r_init[1], r_init[0] * 0.95)
            for phase_shift in [0.0, np.pi/(2*n1), np.pi/n1]:
                if time.time() > deadline - 35:
                    break
                phase_init = [0.0, phase_shift]
                s, p = _optimize_two_ring_config(n1, n2, r_init, phase_init, time_limit=2.5)
                if s > best_overall_score:
                    best_overall_score = s
                    best_overall_points = p
                candidates.append((s, p))
    
    # Phase 2: Coordinate descent polish on top candidates
    # Sort candidates by score, take top few
    candidates.sort(key=lambda x: -x[0])
    top_candidates = candidates[:8] if len(candidates) >= 8 else candidates
    
    for s, p in top_candidates:
        if time.time() > deadline - 25:
            break
        # Soft polish first
        s2, p2 = _coordinate_descent_polish_soft(p, time_limit=8.0, max_rounds=25, beta=20.0)
        if s2 > best_overall_score:
            best_overall_score = s2
            best_overall_points = p2
        # Exact polish
        if time.time() < deadline - 15:
            s3, p3 = _coordinate_descent_polish(p2, time_limit=5.0, max_rounds=15)
            if s3 > best_overall_score:
                best_overall_score = s3
                best_overall_points = p3
    
    # Phase 3: Final Nelder-Mead refinement in full space from best
    if best_overall_points is not None and time.time() < deadline - 10:
        x0 = best_overall_points.flatten()
        def obj(x):
            pts = x.reshape(-1, 2)
            return -_score(pts)
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            s = -res.fun
            if s > best_overall_score:
                best_overall_score = s
                best_overall_points = res.x.reshape(-1, 2).copy()
        except Exception:
            pass
    
    # Fallback: regular polygon
    if best_overall_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_overall_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_overall_score = _score(best_overall_points)
    
    return best_overall_points
