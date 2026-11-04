import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

# Precompute triple indices for vectorized area computation
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])
NUM_TRIPLES = len(TRIPLE_INDICES)

# Precompute which triples involve each point for fast local updates
POINT_TRIPLES = [[] for _ in range(N)]
for idx, (i, j, k) in enumerate(TRIPLE_INDICES):
    POINT_TRIPLES[i].append(idx)
    POINT_TRIPLES[j].append(idx)
    POINT_TRIPLES[k].append(idx)
POINT_TRIPLES = [np.array(pt) for pt in POINT_TRIPLES]


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


def _soft_min_score(points, temperature=0.05):
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
    areas = 0.5 * np.abs(cross)

    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    shifted = areas - min_area
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    soft_min = min_area - temperature * log_sum

    return soft_min / hull_area


def _epigraph_score(points, t, temperature=0.01):
    """
    Epigraph formulation: maximize t subject to all triangle areas >= t.
    We minimize a penalty for violations + soft penalty on t.
    Actually, let's use a different approach: minimize sum of max(0, t - area_i)^2 - t
    This is non-smooth. Let's use soft version.
    """
    if points.shape != (N, 2):
        return 1e6
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 1e6
    except Exception:
        return 1e6

    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    # Penalty for areas below t
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2) / (2 * temperature)
    
    # We want to maximize t, so minimize -t + penalty
    return -t + penalty


def _generate_two_ring(n_out, n_in, r_in, phi_rel):
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + phi_rel
    
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    if n_in > 0:
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        return np.vstack((pts_out, pts_in))
    else:
        return pts_out


def _generate_heptagon_with_offsets(offsets):
    """
    Generate 14 points: 7 on a regular heptagon (outer) and 7 on a regular heptagon (inner),
    where each point has a radial offset from its base position.
    Actually, let's model all 14 points as perturbations of a regular 14-gon or two rings.
    
    Let's use: 7 outer points at angles 2*pi*k/7, radius 1+offsets[k]
    7 inner points at angles 2*pi*k/7 + pi/7, radius 0.5+offsets[k+7]
    """
    n_half = N // 2  # 7
    angles_out = 2 * np.pi * np.arange(n_half) / n_half
    angles_in = angles_out + np.pi / n_half
    
    r_out = 1.0 + offsets[:n_half]
    r_in = 0.5 + offsets[n_half:]
    
    pts_out = np.column_stack((r_out * np.cos(angles_out), r_out * np.sin(angles_out)))
    pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
    
    return np.vstack((pts_out, pts_in))


def _apply_affine(points, a, b, c, d, tx, ty):
    """Apply affine transform: [x'] = [a b; c d] [x] + [tx; ty]"""
    x = points[:, 0]
    y = points[:, 1]
    x_new = a * x + b * y + tx
    y_new = c * x + d * y + ty
    return np.column_stack((x_new, y_new))


def _optimize_heptagon_affine(time_limit=10.0):
    """
    Optimize 7+7 radial offsets + 6 affine parameters (but scale is redundant since we normalize by hull area).
    Actually, affine has 6 params but scaling is absorbed in normalization. Let's use 5: a, b, c, d, tx, ty but fix one scale.
    Let's just optimize the 14 offsets + 6 affine = 20 params. That might be too many.
    
    Alternative: optimize just the 14 radial offsets, then apply best affine.
    Or: parametrize as 14 points = base_heptagon_dual * (1+offsets) + affine.
    
    Let's try: 14 offsets + 3 affine (rotation, shear_x, shear_y) = 17 params.
    """
    start_time = time.time()
    
    # Base configuration: two concentric regular heptagons
    n_half = 7
    angles_out = 2 * np.pi * np.arange(n_half) / n_half
    angles_in = angles_out + np.pi / n_half
    
    def _generate_from_params(params):
        offsets = params[:N]
        rot = params[N]
        shear_x = params[N+1]
        shear_y = params[N+2]
        
        r_out = 1.0 + offsets[:n_half]
        r_in = 0.5 + offsets[n_half:]
        
        pts_out = np.column_stack((r_out * np.cos(angles_out), r_out * np.sin(angles_out)))
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        pts = np.vstack((pts_out, pts_in))
        
        # Apply rotation
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        pts = pts @ R.T
        
        # Apply shear
        S = np.array([[1, shear_x], [shear_y, 1]])
        pts = pts @ S.T
        
        return pts
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        pts = _generate_from_params(params)
        score = _calculate_score(pts)
        return -score
    
    best_score = 0
    best_params = None
    
    # Multiple restarts
    np.random.seed(42)
    for restart in range(5):
        x0 = np.zeros(N + 3)
        if restart > 0:
            x0 = np.random.randn(N + 3) * 0.05
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                         options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500})
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        if time.time() - start_time > time_limit:
            break
    
    if best_params is not None:
        best_points = _generate_from_params(best_params)
    else:
        best_points = _generate_heptagon_with_offsets(np.zeros(N))
        best_score = _calculate_score(best_points)
    
    return best_score, best_points


def _coordinate_descent_polish(points, time_limit=3.0):
    start_time = time.time()
    pts = points.copy()
    
    p_i = pts[TRIPLE_INDICES[:, 0]]
    p_j = pts[TRIPLE_INDICES[:, 1]]
    p_k = pts[TRIPLE_INDICES[:, 2]]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for step in step_sizes:
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(N):
                tri_idx = POINT_TRIPLES[i]
                best_move = None
                best_min_area = min_area
                p_i_curr = pts[i]
                
                # Try 8 directions + opposite
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        new_p = p_i_curr + np.array([dx, dy])
                        
                        # Only update areas for triples involving i
                        min_new = min_area
                        for t_idx in tri_idx:
                            i_t, j_t, k_t = TRIPLE_INDICES[t_idx]
                            p_a = pts[i_t] if i_t != i else new_p
                            p_b = pts[j_t] if j_t != i else new_p
                            p_c = pts[k_t] if k_t != i else new_p
                            v1_t = p_b - p_a
                            v2_t = p_c - p_a
                            cross_t = v1_t[0] * v2_t[1] - v1_t[1] * v2_t[0]
                            area_t = 0.5 * abs(cross_t)
                            if area_t < min_new:
                                min_new = area_t
                                if min_new <= best_min_area:
                                    break
                        
                        if min_new > best_min_area:
                            best_min_area = min_new
                            best_move = new_p
                
                if best_move is not None:
                    pts[i] = best_move
                    min_area = best_min_area
                    improved = True
                    
                    # Recompute all areas
                    p_i = pts[TRIPLE_INDICES[:, 0]]
                    p_j = pts[TRIPLE_INDICES[:, 1]]
                    p_k = pts[TRIPLE_INDICES[:, 2]]
                    v1 = p_j - p_i
                    v2 = p_k - p_i
                    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
                    areas = 0.5 * np.abs(cross)
                    min_area = np.min(areas)
    
    return pts


def _polish_points(points, time_limit=5.0):
    start_time = time.time()
    
    best_pts = points.copy()
    best_score = _calculate_score(points)
    
    # 1. L-BFGS-B on soft-min surrogate
    if time.time() - start_time < time_limit * 0.4:
        temp = 0.05
        def neg_soft_score_flat(x):
            p = x.reshape(N, 2)
            return -_soft_min_score(p, temp)
        
        try:
            res = minimize(neg_soft_score_flat, points.flatten(), method='L-BFGS-B',
                          options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            new_pts = res.x.reshape(N, 2)
            new_score = _calculate_score(new_pts)
            if new_score > best_score:
                best_pts = new_pts
                best_score = new_score
        except Exception:
            pass
    
    # 2. Lower temperature
    if time.time() - start_time < time_limit * 0.7:
        temp2 = 0.01
        def neg_soft_score_flat2(x):
            p = x.reshape(N, 2)
            return -_soft_min_score(p, temp2)
        
        try:
            res2 = minimize(neg_soft_score_flat2, best_pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            new_pts2 = res2.x.reshape(N, 2)
            new_score2 = _calculate_score(new_pts2)
            if new_score2 > best_score:
                best_pts = new_pts2
                best_score = new_score2
        except Exception:
            pass
    
    # 3. Coordinate descent on exact min
    if time.time() - start_time < time_limit * 0.9:
        cd_time = time_limit - (time.time() - start_time) - 0.5
        if cd_time > 0.3:
            best_pts = _coordinate_descent_polish(best_pts, cd_time)
            best_score = _calculate_score(best_pts)
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = 0
    best_points = None
    
    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Strategy 1: Two-ring configurations
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
    
    for n_out, n_in in splits:
        if time.time() > deadline - 20.0:
            break
        
        if n_in == 0:
            angles = np.linspace(0, 2*np.pi, n_out, endpoint=False)
            pts = np.column_stack((np.cos(angles), np.sin(angles)))
        else:
            # Quick two-ring optimization
            best_ring_score = 0
            best_ring_pts = None
            for r0 in [0.3, 0.5, 0.7, 0.9]:
                for phi0 in [0, np.pi/n_in, np.pi/(2*n_in)]:
                    pts_try = _generate_two_ring(n_out, n_in, r0, phi0)
                    s = _calculate_score(pts_try)
                    if s > best_ring_score:
                        best_ring_score = s
                        best_ring_pts = pts_try
            if best_ring_pts is not None:
                pts = best_ring_pts
            else:
                pts = _generate_two_ring(n_out, n_in, 0.5, 0.0)
        
        remaining = deadline - time.time()
        if remaining > 8.0:
            polish_time = min(5.0, remaining * 0.3)
            polished_pts, polished_score = _polish_points(pts, time_limit=polish_time)
            try_update(polished_score, polished_pts)
    
    # Strategy 2: Heptagon with offsets + affine
    if time.time() < deadline - 25.0:
        hept_score, hept_pts = _optimize_heptagon_affine(time_limit=12.0)
        try_update(hept_score, hept_pts)
        
        if time.time() < deadline - 10.0:
            remaining = deadline - time.time() - 3.0
            if remaining > 5.0:
                polished_pts, polished_score = _polish_points(hept_pts, time_limit=remaining)
                try_update(polished_score, polished_pts)
    
    # Final intensive polish on the best points found so far
    if best_points is not None and time.time() < deadline - 8.0:
        remaining = deadline - time.time() - 2.0
        if remaining > 4.0:
            final_pts, final_score = _polish_points(best_points, time_limit=remaining)
            try_update(final_score, final_pts)
    
    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _calculate_score(best_points)

    return best_points
