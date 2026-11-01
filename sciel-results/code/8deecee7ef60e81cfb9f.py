import numpy as np
import time
from scipy.optimize import minimize, basinhopping, differential_evolution
from scipy.spatial import ConvexHull

N = 14

def compute_score(points):
    """Compute min triangle area / convex hull area, vectorized."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # Precompute all C(n,3) triangle indices
    idx = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    p_i = points[idx[:, 0]]
    p_j = points[idx[:, 1]]
    p_k = points[idx[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def neg_score(x):
    points = x.reshape((N, 2))
    return -compute_score(points)

def ring_7_5_2_config(params):
    """
    Parameterize 14 points as 3 concentric regular polygons:
    - 7 points on ring of radius r1 at angles theta1 + 2*pi*k/7
    - 5 points on ring of radius r2 at angles theta2 + 2*pi*k/5
    - 2 points on ring of radius r3 at angles theta3 + 2*pi*k/2
    params: [r1, r2, r3, theta1, theta2, theta3]
    """
    r1, r2, r3, t1, t2, t3 = params
    # Clamp radii to positive
    r1 = max(r1, 0.1)
    r2 = max(r2, 0.05)
    r3 = max(r3, 0.05)
    
    pts = []
    # Ring 1: 7 points
    for k in range(7):
        a = t1 + 2 * np.pi * k / 7
        pts.append([r1 * np.cos(a), r1 * np.sin(a)])
    # Ring 2: 5 points
    for k in range(5):
        a = t2 + 2 * np.pi * k / 5
        pts.append([r2 * np.cos(a), r2 * np.sin(a)])
    # Ring 3: 2 points (diametrically opposite)
    for k in range(2):
        a = t3 + 2 * np.pi * k / 2
        pts.append([r3 * np.cos(a), r3 * np.sin(a)])
    
    return np.array(pts)

def neg_ring_score(params):
    pts = ring_7_5_2_config(params)
    return -compute_score(pts)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(12345)
    
    def try_update(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s
    
    # === Phase 1: Differential Evolution on 6D ring parameter space ===
    # params: [r1, r2, r3, theta1, theta2, theta3]
    bounds = [
        (0.3, 1.5),   # r1
        (0.1, 1.2),   # r2
        (0.05, 1.0),  # r3
        (0, 2*np.pi), # theta1
        (0, 2*np.pi), # theta2
        (0, 2*np.pi), # theta3
    ]
    
    # Multiple DE runs with different seeds and settings
    de_configs = [
        {'maxiter': 150, 'popsize': 30, 'seed': 42, 'polish': False},
        {'maxiter': 100, 'popsize': 40, 'seed': 123, 'polish': False},
        {'maxiter': 80, 'popsize': 50, 'seed': 999, 'polish': False},
    ]
    
    for cfg in de_configs:
        if time.time() - start_time > deadline - 30:
            break
        try:
            res = differential_evolution(
                neg_ring_score, bounds,
                maxiter=cfg['maxiter'],
                popsize=cfg['popsize'],
                seed=cfg['seed'],
                polish=False,
                tol=1e-6,
                init='sobol',
            )
            best_params = res.x
            pts = ring_7_5_2_config(best_params)
            try_update(pts)
        except Exception:
            pass
    
    # === Phase 2: BasinHopping refinement on ring params ===
    if best_points is not None and time.time() - start_time < deadline - 20:
        # Get current best params from the last DE run
        # Re-derive: use the best found params
        # Actually, let's just do basin hopping from a few good starting points
        ring_inits = [
            [1.0, 0.55, 0.3, 0, 0, 0],
            [1.0, 0.6, 0.35, np.pi/7, np.pi/5, np.pi/4],
            [1.2, 0.5, 0.25, 0, np.pi/5, 0],
            [0.9, 0.65, 0.4, np.pi/14, 0, np.pi/8],
            [1.1, 0.45, 0.2, np.pi/7, np.pi/5, 0],
        ]
        
        for init_p in ring_inits:
            if time.time() - start_time > deadline - 15:
                break
            try:
                res = basinhopping(
                    neg_ring_score, np.array(init_p),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10},
                    minsteps=5, T=0.3, stepsize=0.3,
                    interval=4, disp=False,
                )
                pts = ring_7_5_2_config(res.x)
                try_update(pts)
            except Exception:
                pass
    
    # === Phase 3: Full 28D refinement from best ring solution ===
    if best_points is not None and time.time() - start_time < deadline - 25:
        x0 = best_points.flatten()
        try:
            res = basinhopping(
                neg_score, x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9},
                minsteps=3, T=0.5, stepsize=0.3,
                interval=5, disp=False
            )
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    # === Phase 4: Powell polish ===
    if best_points is not None and time.time() - start_time < deadline - 15:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                          options={'maxiter': 400, 'xtol': 1e-8, 'ftol': 1e-10})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    # === Phase 5: More DE with wider bounds if time permits ===
    if time.time() - start_time < deadline - 25:
        bounds_wide = [
            (0.2, 2.0),
            (0.05, 1.5),
            (0.02, 1.2),
            (0, 2*np.pi),
            (0, 2*np.pi),
            (0, 2*np.pi),
        ]
        try:
            res = differential_evolution(
                neg_ring_score, bounds_wide,
                maxiter=120, popsize=35,
                seed=777, polish=False, tol=1e-6,
            )
            pts = ring_7_5_2_config(res.x)
            try_update(pts)
        except Exception:
            pass
    
    # === Phase 6: Final tight polish ===
    if best_points is not None and time.time() - start_time < deadline - 10:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    if best_points is not None and time.time() - start_time < deadline - 5:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                          options={'maxiter': 600, 'xtol': 1e-9, 'ftol': 1e-11})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    return best_points
