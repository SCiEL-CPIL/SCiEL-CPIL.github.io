import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13
# Precompute triangle indices
tri_i = []
tri_j = []
tri_k = []
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            tri_i.append(i)
            tri_j.append(j)
            tri_k.append(k)
tri_i = np.array(tri_i, dtype=np.int32)
tri_j = np.array(tri_j, dtype=np.int32)
tri_k = np.array(tri_k, dtype=np.int32)

def construct():
    start_time = time.time()
    time_limit = 145.0  # 5s margin

    def compute_score(points):
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0, 0.0
        if area_hull < 1e-9:
            return 0.0, 0.0
        x = points[:, 0]
        y = points[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        score = min_area / area_hull
        return score, area_hull

    def soft_min_objective(points_flat, beta):
        points = points_flat.reshape(-1, 2)
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 1e6
        if area_hull < 1e-9:
            return 1e6
        
        x = points[:, 0]
        y = points[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        # Soft min via log-sum-exp
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        return -soft_min_area / area_hull

    def regular_polygon(n, radius=1.0):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

    best_points = regular_polygon(N)
    best_score, _ = compute_score(best_points)

    rng = np.random.RandomState(42)
    
    starts = []
    # Start 1: Perfect regular polygon
    starts.append(regular_polygon(N))
    
    # Start 2: 1 center + 12 on circle (ring structure)
    ring12 = regular_polygon(12, radius=1.0)
    starts.append(np.vstack([[0.0, 0.0], ring12]))
    
    # Start 3: 1 center + 6 inner + 6 outer (two rings)
    inner6 = regular_polygon(6, radius=0.5)
    outer6 = regular_polygon(6, radius=1.0)
    starts.append(np.vstack([[0.0, 0.0], inner6, outer6]))
    
    # Start 4: 1 center + 6 inner + 6 outer with angular offset
    inner6b = regular_polygon(6, radius=0.55)
    outer6b = regular_polygon(6, radius=1.0, radius_offset=np.pi/6) if hasattr(regular_polygon, 'radius_offset') else np.column_stack([1.0 * np.cos(np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6), 1.0 * np.sin(np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6)])
    starts.append(np.vstack([[0.0, 0.0], inner6b, outer6b]))

    # Start 5-8: Angular perturbations of regular polygon
    for i in range(4):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.randn(N) * 0.08
        radii = np.ones(N) + rng.randn(N) * 0.06
        starts.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    # Start 9-12: Larger perturbations
    for i in range(4):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.randn(N) * 0.25
        radii = np.ones(N) + rng.randn(N) * 0.12
        starts.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))

    for start_pts in starts:
        if time.time() > start_time + time_limit - 15:
            break
        
        current = start_pts.copy()
        
        # Annealing with Soft-Min: gradually increase beta
        betas = [5, 20, 100, 500, 2000]
        for beta in betas:
            if time.time() > start_time + time_limit - 15:
                break
            try:
                res = opt.minimize(
                    lambda x: soft_min_objective(x, beta),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                if res.fun < 1e5:
                    current = res.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Final high-precision polish: use very high beta soft-min (close to hard min)
        if time.time() < start_time + time_limit - 10:
            try:
                res = opt.minimize(
                    lambda x: soft_min_objective(x, 10000),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50}
                )
                if res.fun < 1e5:
                    candidate = res.x.reshape(-1, 2)
                    s, _ = compute_score(candidate)
                    if s > best_score:
                        best_score = s
                        best_points = candidate.copy()
            except Exception:
                pass
        
        # Additional Nelder-Mead polish on hard objective if time permits
        if time.time() < start_time + time_limit - 5:
            try:
                def hard_obj(x):
                    pts = x.reshape(-1, 2)
                    sc, _ = compute_score(pts)
                    return -sc
                res = opt.minimize(
                    hard_obj,
                    current.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                candidate = res.x.reshape(-1, 2)
                s, _ = compute_score(candidate)
                if s > best_score:
                    best_score = s
                    best_points = candidate.copy()
            except Exception:
                pass

    return best_points
