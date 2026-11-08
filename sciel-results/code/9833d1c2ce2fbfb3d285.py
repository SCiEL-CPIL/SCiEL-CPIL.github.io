"""
Heilbronn convex regions n=14

Strategy:
1. Use structured ring-based parameterization: points on 1-3 concentric rings
   with optimized radii and angular offsets.

2. Use epigraph formulation: maximize t subject to all triangle areas >= t
   and hull area = 1 (via scaling).

3. Coordinate descent with exact evaluation for fast local optimization.

4. Multiple restarts with different ring structures and random perturbations.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 148.0  # Safety margin before 150s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3 = 364)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)


def _min_triangle_area(points):
    """Compute minimum triangle area."""
    areas = _all_triangle_areas(points)
    return float(np.min(areas))


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_a


def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(-1, 2)
    s = _score(pts)
    return -s


def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts


def _ring_to_pts(n, ring_config):
    """
    Convert ring configuration to points.
    ring_config: list of (n_points, radius, angle_offset)
    """
    pts_list = []
    for n_r, radius, offset in ring_config:
        angles = 2 * np.pi * np.arange(n_r) / n_r + offset
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        pts_list.append(np.column_stack([x, y]))
    return np.vstack(pts_list)


def _initial_single_ring(n, seed=0):
    """Regular n-gon."""
    rng = np.random.RandomState(seed)
    offset = rng.uniform(0, 2 * np.pi)
    config = [(n, 1.0, offset)]
    return _normalize_unit_area(_ring_to_pts(n, config))


def _initial_two_ring(n, seed=0):
    """Two rings: 10 outer + 4 inner."""
    rng = np.random.RandomState(seed)
    outer_n = 10
    inner_n = n - outer_n
    r_inner = rng.uniform(0.3, 0.7)
    outer_offset = rng.uniform(0, 2 * np.pi)
    inner_offset = rng.uniform(0, 2 * np.pi)
    config = [(outer_n, 1.0, outer_offset), (inner_n, r_inner, inner_offset)]
    return _normalize_unit_area(_ring_to_pts(n, config))


def _initial_three_ring(n, seed=0):
    """Three rings: 8 outer + 4 middle + 2 inner."""
    rng = np.random.RandomState(seed)
    r_mid = rng.uniform(0.5, 0.75)
    r_inner = rng.uniform(0.2, 0.4)
    offset1 = rng.uniform(0, 2 * np.pi)
    offset2 = rng.uniform(0, 2 * np.pi)
    offset3 = rng.uniform(0, 2 * np.pi)
    config = [(8, 1.0, offset1), (4, r_mid, offset2), (2, r_inner, offset3)]
    return _normalize_unit_area(_ring_to_pts(n, config))


def _coordinate_descent(pts, time_budget, step_size=0.05, max_iter=5000):
    """
    Coordinate descent: optimize each point's x and y separately.
    Uses exact score evaluation.
    """
    start_time = time.time()
    best_pts = pts.copy()
    best_score = _score(best_pts)
    current_pts = pts.copy()
    current_score = best_score
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        for i in range(N):
            # Optimize x coordinate
            x_best = current_pts[i, 0]
            s_best = current_score
            
            # Try several steps in x
            for dx in [-step_size, -step_size/2, step_size/2, step_size]:
                trial = current_pts.copy()
                trial[i, 0] += dx
                s_trial = _score(trial)
                if s_trial > s_best:
                    s_best = s_trial
                    x_best = trial[i, 0]
            
            # Optimize y coordinate
            y_best = x_best  # placeholder, will fix
            current_pts[i, 0] = x_best
            
            s_after_x = _score(current_pts)
            s_best_y = s_after_x
            y_best = current_pts[i, 1]
            
            for dy in [-step_size, -step_size/2, step_size/2, step_size]:
                trial = current_pts.copy()
                trial[i, 1] += dy
                s_trial = _score(trial)
                if s_trial > s_best_y:
                    s_best_y = s_trial
                    y_best = trial[i, 1]
            
            current_pts[i, 1] = y_best
            current_score = _score(current_pts)
            
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
                improved = True
        
        # Adaptive step size
        if improved:
            step_size = max(0.01, step_size * 0.8)
        else:
            step_size = min(0.2, step_size * 1.2)
    
    return best_pts, best_score


def _nelder_mead_polish(pts, time_budget):
    """
    Nelder-Mead polish on the full 28-dimensional space.
    """
    start_time = time.time()
    pts_flat = pts.flatten()
    
    def obj_func(p):
        if time.time() - start_time > time_budget:
            raise TimeoutError()
        return _neg_score(p)
    
    try:
        res = minimize(obj_func, pts_flat, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                              'adaptive': True, 'maxfev': 6000})
        if res.x is not None:
            cand = res.x.reshape(-1, 2)
            s = _score(cand)
            return cand, s
    except Exception:
        pass
    
    return pts, _score(pts)


def construct():
    start_time = time.time()
    
    # Generate diverse initial configurations
    inits = []
    # Single ring
    for seed in range(5):
        inits.append(_initial_single_ring(N, seed))
    # Two rings
    for seed in range(5):
        inits.append(_initial_two_ring(N, seed))
    # Three rings
    for seed in range(5):
        inits.append(_initial_three_ring(N, seed))
    
    # Evaluate initial scores
    init_scores = []
    for init in inits:
        s = _score(init)
        init_scores.append(s)
    
    # Sort by initial score (best first)
    sorted_indices = np.argsort(init_scores)[::-1]
    inits = [inits[i] for i in sorted_indices]
    init_scores = [init_scores[i] for i in sorted_indices]
    
    best_pts = inits[0].copy()
    best_score = init_scores[0]
    
    # Phase 1: Coordinate descent on top 5 initial configs
    # Allocate time: ~80s for coordinate descent, ~60s for NM polish
    cd_total_budget = 80.0
    cd_per_config = cd_total_budget / 5
    
    for i in range(5):
        if time.time() - start_time > DEADLINE - 60:
            break
        
        cd_budget = min(cd_per_config, DEADLINE - 60 - (time.time() - start_time))
        if cd_budget < 2:
            break
        
        try:
            cand_pts, cand_score = _coordinate_descent(inits[i], cd_budget)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand_pts
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish on best result
    if time.time() - start_time < DEADLINE - 10:
        nm_budget = min(50.0, DEADLINE - 10 - (time.time() - start_time))
        if nm_budget > 5:
            try:
                cand_pts, cand_score = _nelder_mead_polish(best_pts, nm_budget)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
            except Exception:
                pass
    
    # Phase 3: Final coordinate descent polish with smaller steps
    if time.time() - start_time < DEADLINE - 3:
        final_budget = min(10.0, DEADLINE - 3 - (time.time() - start_time))
        if final_budget > 2:
            try:
                cand_pts, cand_score = _coordinate_descent(best_pts, final_budget, step_size=0.01, max_iter=2000)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = _initial_single_ring(N)
    
    return best_pts
