import numpy as np
import time
import itertools
import scipy.optimize as opt
import scipy.spatial as spat
import warnings

N = 13
DEADLINE = 145.0  # Safety margin before 150s kill
START_TIME = time.time()

# Precompute triangle indices
TRI_INDICES = np.array(list(itertools.combinations(range(N), 3)))
NUM_TRIS = len(TRI_INDICES)

def _get_triangle_areas(points):
    """Compute areas of all C(13,3)=286 triangles."""
    p1 = points[TRI_INDICES[:, 0]]
    p2 = points[TRI_INDICES[:, 1]]
    p3 = points[TRI_INDICES[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    """Compute convex hull area, return 0 on failure."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Minimum triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0
    areas = _get_triangle_areas(points)
    return np.min(areas) / hull_area

def _softmin_score(points, beta=50.0):
    """Smooth approximation of min triangle area / hull area using soft-min."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0
    areas = _get_triangle_areas(points)
    # Soft-min: -1/beta * log(sum(exp(-beta*a_i))) approximates min(a_i)
    # Use log-sum-exp for numerical stability
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
    softmin = min_a - (1.0/beta) * log_sum_exp
    return softmin / hull_area

def _neg_softmin_score(x, beta=50.0):
    """Objective for minimization: negative softmin score."""
    pts = x.reshape(N, 2)
    return -_softmin_score(pts, beta)

def _neg_score(x):
    """Objective for minimization: negative exact score."""
    pts = x.reshape(N, 2)
    return -_score(pts)

def _normalize_hull_area(points):
    """Scale points so convex hull area is 1."""
    area = _hull_area(points)
    if area > 1e-9:
        scale = 1.0 / np.sqrt(area)
        return points * scale
    return points

def _make_ring_config(n_inner, n_outer, r_inner=0.4, r_outer=1.0, offset_inner=0.0, offset_outer=0.0):
    """Create configuration with center + inner ring + outer ring."""
    pts = np.zeros((N, 2))
    idx = 0
    pts[idx] = [0, 0]; idx += 1
    if n_inner > 0:
        angles_in = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + offset_inner
        pts[idx:idx+n_inner] = np.column_stack([r_inner*np.cos(angles_in), r_inner*np.sin(angles_in)])
        idx += n_inner
    if n_outer > 0:
        angles_out = np.linspace(0, 2*np.pi, n_outer, endpoint=False) + offset_outer
        pts[idx:idx+n_outer] = np.column_stack([r_outer*np.cos(angles_out), r_outer*np.sin(angles_out)])
        idx += n_outer
    # If we have leftover points, place them on outer ring
    if idx < N:
        remaining = N - idx
        angles_rem = np.linspace(0, 2*np.pi, remaining, endpoint=False) + 0.3
        pts[idx:idx+remaining] = np.column_stack([r_outer*1.1*np.cos(angles_rem), r_outer*1.1*np.sin(angles_rem)])
    return _normalize_hull_area(pts)

def _generate_diverse_configs():
    """Generate diverse initial configurations."""
    configs = []
    # Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(_normalize_hull_area(pts))
    
    # 1 center + 12 ring
    configs.append(_make_ring_config(0, 12, r_outer=1.0))
    
    # 1 center + 6 inner + 6 outer
    configs.append(_make_ring_config(6, 6, r_inner=0.45, r_outer=1.0, offset_inner=np.pi/6))
    
    # 1 center + 4 inner + 8 outer
    configs.append(_make_ring_config(4, 8, r_inner=0.4, r_outer=1.0, offset_inner=np.pi/4, offset_outer=np.pi/8))
    
    # 1 center + 8 inner + 4 outer
    configs.append(_make_ring_config(8, 4, r_inner=0.5, r_outer=1.0, offset_inner=0.0, offset_outer=np.pi/4))
    
    # 1 center + 3+3+6 (two inner rings + outer)
    pts = np.zeros((N, 2))
    pts[0] = [0, 0]
    angles1 = np.linspace(0, 2*np.pi, 3, endpoint=False)
    pts[1:4] = np.column_stack([0.35*np.cos(angles1), 0.35*np.sin(angles1)])
    angles2 = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
    pts[4:7] = np.column_stack([0.65*np.cos(angles2), 0.65*np.sin(angles2)])
    angles3 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    pts[7:13] = np.column_stack([1.0*np.cos(angles3), 1.0*np.sin(angles3)])
    configs.append(_normalize_hull_area(pts))
    
    # 1 center + 5+7
    configs.append(_make_ring_config(5, 7, r_inner=0.5, r_outer=1.0, offset_inner=np.pi/5))
    
    # 1 center + 2+4+6
    pts = np.zeros((N, 2))
    pts[0] = [0, 0]
    angles1 = np.linspace(0, 2*np.pi, 2, endpoint=False)
    pts[1:3] = np.column_stack([0.3*np.cos(angles1), 0.3*np.sin(angles1)])
    angles2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
    pts[3:7] = np.column_stack([0.6*np.cos(angles2), 0.6*np.sin(angles2)])
    angles3 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    pts[7:13] = np.column_stack([1.0*np.cos(angles3), 1.0*np.sin(angles3)])
    configs.append(_normalize_hull_area(pts))
    
    return configs

def _local_search(points, step_size=0.05, max_iters=200, beta=50.0):
    """Coordinate-wise local search with adaptive step size."""
    best_pts = points.copy()
    best_score = _score(best_pts)
    current_step = step_size
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            for d in range(2):
                # Try positive and negative steps
                for direction in [1, -1]:
                    trial = best_pts.copy()
                    trial[i, d] += direction * current_step
                    trial_score = _score(trial)
                    if trial_score > best_score:
                        best_pts = trial
                        best_score = trial_score
                        improved = True
        if not improved:
            # Reduce step size
            current_step *= 0.7
            if current_step < 1e-5:
                break
        else:
            # Slightly increase step size if improving
            current_step *= 1.1
    
    return best_pts, best_score

def _optimize_from_start(init_pts, time_budget=30.0):
    """Multi-stage optimization from a single initial configuration."""
    start_time = time.time()
    x0 = init_pts.flatten()
    
    # Stage 1: Soft-min optimization with Nelder-Mead (smooth objective)
    try:
        res1 = opt.minimize(
            _neg_softmin_score,
            x0,
            method='Nelder-Mead',
            args=(50.0,),
            options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        if time.time() - start_time > time_budget * 0.6:
            return res1.x.reshape(N, 2)
        
        best_pts = res1.x.reshape(N, 2)
    except Exception:
        best_pts = init_pts.copy()
    
    # Stage 2: Refine with higher beta soft-min
    try:
        res2 = opt.minimize(
            _neg_softmin_score,
            best_pts.flatten(),
            method='Nelder-Mead',
            args=(100.0,),
            options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7}
        )
        if time.time() - start_time > time_budget * 0.8:
            return res2.x.reshape(N, 2)
        
        candidate = res2.x.reshape(N, 2)
        if _score(candidate) > _score(best_pts):
            best_pts = candidate
    except Exception:
        pass
    
    # Stage 3: Coordinate-wise local search
    if time.time() - start_time < time_budget:
        try:
            best_pts, _ = _local_search(best_pts, step_size=0.03, max_iters=100)
        except Exception:
            pass
    
    return best_pts

def construct():
    global START_TIME
    if 'START_TIME' not in globals() or START_TIME is None:
        START_TIME = time.time()
    
    current_time = time.time()
    elapsed = current_time - START_TIME
    time_left = DEADLINE - elapsed
    
    if time_left < 15:
        configs = _generate_diverse_configs()
        best_pts = configs[0]
        for cfg in configs:
            if _score(cfg) > _score(best_pts):
                best_pts = cfg
        return best_pts
    
    configs = _generate_diverse_configs()
    
    # Score all initial configs
    config_scores = [_score(cfg) for cfg in configs]
    best_score = max(config_scores)
    best_pts = configs[np.argmax(config_scores)]
    
    # Sort configs by score (best first)
    sorted_idx = np.argsort(config_scores)[::-1]
    
    # Allocate time budget per config
    n_configs = len(configs)
    time_per_config = max(10.0, (time_left - 15.0) / n_configs)
    
    for i, idx in enumerate(sorted_idx):
        current_time = time.time()
        elapsed = current_time - START_TIME
        time_left = DEADLINE - elapsed
        
        if time_left < 15:
            break
        
        init_pts = configs[idx]
        budget = min(time_per_config, time_left - 15.0)
        
        if budget < 5:
            break
        
        try:
            result_pts = _optimize_from_start(init_pts, time_budget=budget)
            result_score = _score(result_pts)
            
            if result_score > best_score:
                best_score = result_score
                best_pts = result_pts
        except Exception:
            continue
    
    return best_pts
