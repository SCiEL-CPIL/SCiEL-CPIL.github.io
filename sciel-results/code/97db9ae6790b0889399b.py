import numpy as np
import time
import itertools
import scipy.optimize as opt
import scipy.spatial as spat
import warnings

warnings.filterwarnings('ignore')

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
        # Ensure at least 3 non-collinear points for ConvexHull
        if len(points) < 3:
            return 0.0
        # Check for collinearity/degeneracy
        if np.allclose(points, points[0]):
            return 0.0
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

def _optimize_angular(angles_init, deadline):
    """
    Optimize points constrained to the unit circle by optimizing their angles.
    This ensures convexity and reduces dimensionality.
    """
    def _angle_objective(theta):
        # theta is array of N angles
        # Ensure angles are within [0, 2pi]
        theta = np.mod(theta, 2*np.pi)
        # Sort angles to ensure consistent hull ordering? 
        # ConvexHull handles ordering internally, but for triangle area calc, order doesn't matter.
        # However, if we want to enforce "convex position", sorting helps visualization but not math.
        pts = np.column_stack([np.cos(theta), np.sin(theta)])
        # Check for degeneracy (points too close)
        if np.any(np.diff(np.sort(theta)) < 1e-4):
            return 1e6 # Penalty for collision
        return -_softmin_score(pts, beta=50.0)
    
    # Use SLSQP for bounds
    bounds = [(0, 2*np.pi) for _ in range(N)]
    
    try:
        res = opt.minimize(
            _angle_objective,
            angles_init,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        if res.success:
            theta_opt = np.mod(res.x, 2*np.pi)
            pts = np.column_stack([np.cos(theta_opt), np.sin(theta_opt)])
            return pts
    except Exception:
        pass
    
    # Fallback to Nelder-Mead if SLSQP fails
    try:
        res = opt.minimize(
            _angle_objective,
            angles_init,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        theta_opt = np.mod(res.x, 2*np.pi)
        pts = np.column_stack([np.cos(theta_opt), np.sin(theta_opt)])
        return pts
    except Exception:
        return np.column_stack([np.cos(angles_init), np.sin(angles_init)])

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
        best_score = _score(best_pts)
        for cfg in configs:
            s = _score(cfg)
            if s > best_score:
                best_score = s
                best_pts = cfg
        return best_pts
    
    best_score = 0.0
    best_pts = None
    
    # 1. Angular Optimization Phase
    # Generate several angular starting points
    angular_starts = []
    # Regular polygon
    angular_starts.append(np.linspace(0, 2*np.pi, N, endpoint=False))
    # Jittered regular polygon
    rng = np.random.RandomState(42)
    for _ in range(3):
        jitter = rng.uniform(-0.1, 0.1, N)
        angular_starts.append(np.linspace(0, 2*np.pi, N, endpoint=False) + jitter)
    
    angular_candidates = []
    for start_angles in angular_starts:
        if time.time() > DEADLINE - 30:
            break
        try:
            pts = _optimize_angular(start_angles, DEADLINE)
            pts = _normalize_hull_area(pts)
            angular_candidates.append(pts)
        except Exception:
            continue
    
    # 2. General Optimization Phase
    # Combine angular candidates with diverse ring configs
    general_configs = _generate_diverse_configs()
    all_starts = angular_candidates + general_configs
    
    # Sort by initial score to prioritize good starts
    scored_starts = []
    for cfg in all_starts:
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_pts = cfg
        scored_starts.append((s, cfg))
    
    scored_starts.sort(key=lambda x: x[0], reverse=True)
    
    # Allocate time budget
    n_starts = len(scored_starts)
    time_per_config = max(5.0, (time_left - 20.0) / max(1, n_starts))
    
    for score, cfg in scored_starts:
        if time.time() > DEADLINE - 10:
            break
        
        budget = min(time_per_config, time_left - 10.0)
        if budget < 2:
            break
            
        try:
            result_pts = _optimize_from_start(cfg, time_budget=budget)
            result_score = _score(result_pts)
            
            if result_score > best_score:
                best_score = result_score
                best_pts = result_pts
        except Exception:
            continue
    
    if best_pts is None:
        # Fallback
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_pts
