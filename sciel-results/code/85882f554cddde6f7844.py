import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    if points.shape != (N, 2):
        return 0.0
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_dist:
                min_dist = d
    if min_dist < 1e-12:
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

def _score_two_ring(params, n_out, n_in):
    r_in, theta_in = params
    if r_in < 0.05 or r_in > 0.99:
        return 0.0
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta_in
    points = np.vstack([
        np.column_stack((np.cos(angles_out), np.sin(angles_out))),
        np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
    ])
    return _calculate_score(points)

def _score_three_ring(params, n1, n2, n3):
    r1, r2, th1, th2 = params
    if r1 < 0.05 or r1 > 0.7 or r2 < 0.5 or r2 > 0.99:
        return 0.0
    a1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
    a2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + th1
    a3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + th2
    pts1 = np.column_stack((np.cos(a1), np.sin(a1)))
    pts2 = np.column_stack((r1 * np.cos(a2), r1 * np.sin(a2)))
    pts3 = np.column_stack((r2 * np.cos(a3), r2 * np.sin(a3)))
    points = np.vstack([pts1, pts2, pts3])
    return _calculate_score(points)

def _optimize_general(points, time_limit):
    start_time = time.time()
    best_score = _calculate_score(points)
    best_points = points.copy()
    x0 = points.flatten()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return 1e9
        pts = x.reshape(N, 2)
        return -_calculate_score(pts)
    
    try:
        res = minimize(
            objective, x0, method='Nelder-Mead',
            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000, 'maxfev': 5000, 'disp': False}
        )
        if time.time() - start_time < time_limit:
            new_points = res.x.reshape(N, 2)
            new_score = _calculate_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
    except Exception:
        pass
    return best_score, best_points

def _optimize_two_ring(n_out, n_in, time_limit):
    start_time = time.time()
    best_score = 0.0
    best_params = None
    
    def obj(params):
        if time.time() - start_time > time_limit:
            return 1e9
        return -_score_two_ring(params, n_out, n_in)
    
    try:
        res = minimize(
            obj, [0.5, 0.0], method='Nelder-Mead',
            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500, 'disp': False}
        )
        if res.fun < 0:
            best_params = res.x
            best_score = -res.fun
    except Exception:
        pass
    
    if best_params is not None and best_score > 0:
        r_in, theta_in = best_params
        angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
        angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta_in
        points = np.vstack([
            np.column_stack((np.cos(angles_out), np.sin(angles_out))),
            np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        ])
        return best_score, points
    return 0.0, None

def _optimize_three_ring(n1, n2, n3, time_limit):
    start_time = time.time()
    best_score = 0.0
    best_params = None
    
    def obj(params):
        if time.time() - start_time > time_limit:
            return 1e9
        return -_score_three_ring(params, n1, n2, n3)
    
    try:
        res = minimize(
            obj, [0.3, 0.7, 0.0, 0.0], method='Nelder-Mead',
            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500, 'disp': False}
        )
        if res.fun < 0:
            best_params = res.x
            best_score = -res.fun
    except Exception:
        pass
    
    if best_params is not None and best_score > 0:
        r1, r2, th1, th2 = best_params
        a1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + th1
        a3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + th2
        pts1 = np.column_stack((np.cos(a1), np.sin(a1)))
        pts2 = np.column_stack((r1 * np.cos(a2), r1 * np.sin(a2)))
        pts3 = np.column_stack((r2 * np.cos(a3), r2 * np.sin(a3)))
        points = np.vstack([pts1, pts2, pts3])
        return best_score, points
    return 0.0, None

def _generate_regular_polygon():
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _generate_two_ring_config(n_out, n_in, r_in, theta_in):
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta_in
    return np.vstack([
        np.column_stack((np.cos(angles_out), np.sin(angles_out))),
        np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
    ])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    best_score = 0.0
    best_points = None
    
    # 1. Regular polygon baseline
    reg = _generate_regular_polygon()
    score, pts = _optimize_general(reg, 15.0)
    if score > best_score:
        best_score = score
        best_points = pts
    
    # 2. Two-ring configurations with different splits
    two_ring_configs = [
        (7, 7, 0.5, 0.0),
        (7, 7, 0.6, np.pi/7),
        (7, 7, 0.4, np.pi/7),
        (6, 8, 0.5, 0.0),
        (8, 6, 0.5, 0.0),
        (7, 7, 0.7, np.pi/7),
    ]
    
    for n_out, n_in, r_in, theta_in in two_ring_configs:
        if time.time() > deadline - 30:
            break
        pts = _generate_two_ring_config(n_out, n_in, r_in, theta_in)
        score, opt_pts = _optimize_general(pts, 12.0)
        if score > best_score:
            best_score = score
            best_points = opt_pts
    
    # 3. Dedicated two-ring parameter optimization
    for n_out, n_in in [(7, 7), (6, 8), (8, 6)]:
        if time.time() > deadline - 25:
            break
        score, pts = _optimize_two_ring(n_out, n_in, 10.0)
        if score > best_score and pts is not None:
            best_score = score
            best_points = pts
            if time.time() < deadline - 10:
                pol_score, pol_pts = _optimize_general(pts, 8.0)
                if pol_score > best_score:
                    best_score = pol_score
                    best_points = pol_pts
    
    # 4. Three-ring configurations
    three_ring_configs = [
        (5, 5, 4, 0.3, 0.7, 0.0, 0.0),
        (5, 5, 4, 0.4, 0.75, np.pi/5, np.pi/4),
        (4, 5, 5, 0.3, 0.7, 0.0, 0.0),
        (6, 4, 4, 0.35, 0.7, 0.0, 0.0),
        (4, 4, 6, 0.35, 0.7, 0.0, 0.0),
    ]
    
    for n1, n2, n3, r1, r2, th1, th2 in three_ring_configs:
        if time.time() > deadline - 25:
            break
        a1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + th1
        a3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + th2
        pts1 = np.column_stack((np.cos(a1), np.sin(a1)))
        pts2 = np.column_stack((r1 * np.cos(a2), r1 * np.sin(a2)))
        pts3 = np.column_stack((r2 * np.cos(a3), r2 * np.sin(a3)))
        points = np.vstack([pts1, pts2, pts3])
        score, opt_pts = _optimize_general(points, 10.0)
        if score > best_score:
            best_score = score
            best_points = opt_pts
    
    # 5. Dedicated three-ring parameter optimization
    for n1, n2, n3 in [(5, 5, 4), (6, 4, 4), (4, 5, 5)]:
        if time.time() > deadline - 20:
            break
        score, pts = _optimize_three_ring(n1, n2, n3, 8.0)
        if score > best_score and pts is not None:
            best_score = score
            best_points = pts
            if time.time() < deadline - 8:
                pol_score, pol_pts = _optimize_general(pts, 6.0)
                if pol_score > best_score:
                    best_score = pol_score
                    best_points = pol_pts
    
    # 6. Final polish if time remains
    if best_points is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 3:
            pol_score, pol_pts = _optimize_general(best_points, remaining)
            if pol_score > best_score:
                best_score = pol_score
                best_points = pol_pts
    
    if best_points is None:
        best_points = reg
    
    return best_points
