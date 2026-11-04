import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

# Precompute triple indices for vectorized area computation
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
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

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    min_area = 0.5 * np.min(np.abs(cross))
    return min_area / hull_area

def _soft_min_value(areas, temp):
    max_a = np.max(areas)
    if max_a < 1e-12:
        return 0.0
    shifted = areas - max_a
    log_sum = np.log(np.sum(np.exp(shifted / temp)))
    return max_a - temp * log_sum

def _smooth_objective(x, temp=0.05):
    p = x.reshape(N, 2)
    if p.shape != (N, 2):
        return 1e6
    try:
        hull = ConvexHull(p)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 1e6
    except Exception:
        return 1e6

    p_i = p[TRIPLE_INDICES[:, 0]]
    p_j = p[TRIPLE_INDICES[:, 1]]
    p_k = p[TRIPLE_INDICES[:, 2]]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_area_soft = _soft_min_value(areas, temp)
    return -min_area_soft / hull_area

def _generate_two_ring(n_out, n_in, r_in, phi_rel):
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + phi_rel
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    if n_in > 0:
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        return np.vstack((pts_out, pts_in))
    else:
        return pts_out

def _optimize_two_ring(n_out, n_in, time_limit=6.0):
    start_time = time.time()
    
    def objective(params):
        r_in, phi_rel = params
        if r_in < 0 or r_in > 1.5:
            return 1e6
        pts = _generate_two_ring(n_out, n_in, r_in, phi_rel)
        score = _calculate_score(pts)
        return -score

    best_score = 0
    best_params = None
    
    start_r = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    start_phi = [0, np.pi/7, np.pi/6, np.pi/5, np.pi/4, np.pi/3]
    
    for r0 in start_r:
        for phi0 in start_phi:
            if time.time() - start_time > time_limit:
                return best_score, best_params
            x0 = [r0, phi0]
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                             options={'xatol':1e-8, 'fatol':1e-8, 'maxiter':500})
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    return best_score, best_params

def _fast_coordinate_descent(points, time_limit=3.0):
    """Fast coordinate descent with adaptive steps and multiple directions."""
    start_time = time.time()
    pts = points.copy()
    score = _calculate_score(pts)
    
    base_steps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
    
    for step in base_steps:
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(N):
                # 8 directions: 4 cardinal + 4 diagonal
                dirs = [[step,0],[-step,0],[0,step],[0,-step],
                        [step,step],[-step,step],[step,-step],[-step,-step]]
                for dx, dy in dirs:
                    trial = pts.copy()
                    trial[i] += [dx, dy]
                    trial_score = _calculate_score(trial)
                    if trial_score > score:
                        pts = trial
                        score = trial_score
                        improved = True
                        break
                if time.time() - start_time > time_limit:
                    return pts
        if not improved:
            break
    
    return pts

def _polish_points(points, time_limit=8.0):
    """Multi-stage polishing: smooth gradient + fast coordinate descent."""
    start_time = time.time()
    best_pts = points.copy()
    best_score = _calculate_score(points)
    
    # Phase 1: Smooth gradient-based optimization with adaptive temperature
    temps = [0.1, 0.05, 0.02, 0.01, 0.005]
    for temp in temps:
        if time.time() - start_time > time_limit * 0.5:
            break
        try:
            res = minimize(lambda x: _smooth_objective(x, temp), 
                          best_pts.flatten(), method='L-BFGS-B',
                          options={'maxiter':300, 'ftol':1e-15, 'gtol':1e-13, 'maxls':200})
            new_pts = res.x.reshape(N, 2)
            new_score = _calculate_score(new_pts)
            if new_score > best_score:
                best_pts = new_pts
                best_score = new_score
        except Exception:
            pass
    
    # Phase 2: Fast coordinate descent with fine steps
    if time.time() - start_time < time_limit * 0.85:
        remaining = time_limit - (time.time() - start_time)
        polished = _fast_coordinate_descent(best_pts, time_limit=max(1.0, remaining * 0.7))
        polished_score = _calculate_score(polished)
        if polished_score > best_score:
            best_pts = polished
            best_score = polished_score
    
    # Phase 3: Final ultra-fine coordinate descent
    if time.time() - start_time < time_limit * 0.95:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 0.5:
            pts = best_pts.copy()
            score = best_score
            for step in [0.0001, 0.00005, 0.00002, 0.00001]:
                if time.time() - start_time > time_limit:
                    break
                improved = True
                while improved and time.time() - start_time < time_limit:
                    improved = False
                    for i in range(N):
                        dirs = [[step,0],[-step,0],[0,step],[0,-step],
                                [step,step],[-step,step],[step,-step],[-step,-step]]
                        for dx, dy in dirs:
                            trial = pts.copy()
                            trial[i] += [dx, dy]
                            trial_score = _calculate_score(trial)
                            if trial_score > score:
                                pts = trial
                                score = trial_score
                                improved = True
                                break
                        if time.time() - start_time > time_limit:
                            break
                    if time.time() - start_time > time_limit:
                        break
            if score > best_score:
                best_pts = pts
                best_score = score
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_score = 0
    best_points = None
    
    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Two-ring configurations to try
    splits = [
        (14, 0),
        (13, 1),
        (12, 2),
        (11, 3),
        (10, 4),
        (9, 5),
        (8, 6),
        (7, 7),
    ]
    
    # Time allocation: reserve ~35s for final polish
    for n_out, n_in in splits:
        if time.time() > deadline - 40:
            break
        
        if n_in == 0:
            pts = _generate_two_ring(n_out, 0, 0, 0)
            remaining = deadline - time.time() - 35
            polish_time = max(3.0, min(8.0, remaining))
            polished_pts, polished_score = _polish_points(pts, time_limit=polish_time)
            try_update(polished_score, polished_pts)
            continue
        
        remaining_for_ring = deadline - time.time() - 40
        ring_time = max(4.0, min(8.0, remaining_for_ring / 2.0))
        score, params = _optimize_two_ring(n_out, n_in, time_limit=ring_time)
        if params is not None:
            pts = _generate_two_ring(n_out, n_in, params[0], params[1])
            remaining = deadline - time.time() - 35
            polish_time = max(3.0, min(8.0, remaining))
            polished_pts, polished_score = _polish_points(pts, time_limit=polish_time)
            try_update(polished_score, polished_pts)
    
    # Also try some non-ring symmetric starting points
    if time.time() < deadline - 40:
        # Regular 14-gon with slight perturbations
        rng = np.random.RandomState(123)
        for trial in range(3):
            if time.time() > deadline - 40:
                break
            angles = np.linspace(0, 2*np.pi, N, endpoint=False) + rng.randn(N) * 0.05
            radii = np.ones(N) + rng.randn(N) * 0.05
            pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
            remaining = deadline - time.time() - 35
            polish_time = max(3.0, min(6.0, remaining))
            polished_pts, polished_score = _polish_points(pts, time_limit=polish_time)
            try_update(polished_score, polished_pts)
    
    # Final intensive polish on best points
    if best_points is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        final_pts, final_score = _polish_points(best_points, time_limit=max(10.0, remaining))
        try_update(final_score, final_pts)
    
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        remaining = deadline - time.time() - 3
        polished_pts, polished_score = _polish_points(best_points, time_limit=max(10.0, remaining))
        best_points = polished_pts
    
    return best_points
