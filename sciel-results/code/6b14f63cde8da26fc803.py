"""
Combined approach: Multi-start local search with ring-parameterized search space,
soft-min objective for gradient-based methods, and differential evolution for global search.
Structural change: parameterize points on concentric rings to reduce dimensionality
and exploit the known structure of good configurations.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by triples of points.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dy_ji * dx_ki
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min_triangle_area / hull_area.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _all_areas(points):
    """
    Calculate all triangle areas for the soft-min objective.
    """
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dy_ji * dx_ki
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

def _objective_hard(flat):
    """
    Hard objective: minimize hull_area / min_area.
    """
    pts = flat.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
    return hull_area / min_area

def _objective_soft(flat, beta=50.0):
    """
    Soft objective using log-sum-exp approximation of min area.
    Provides smoother gradients for L-BFGS-B.
    """
    pts = flat.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    areas = _all_areas(pts)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    # Soft minimum via log-sum-exp
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min

def _decode_ring_params(params, ring_split=(7, 7)):
    """
    Decode ring parameters to points.
    params: [r1, a1_0, ..., a1_{k1-1}, r2, a2_0, ..., a2_{k2-1}, ...]
    """
    pts = np.zeros((N, 2))
    idx = 0
    for k, count in enumerate(ring_split):
        r = params[idx]
        idx += 1
        for i in range(count):
            angle = params[idx]
            idx += 1
            pts[k * 7 + i] = [r * np.cos(angle), r * np.sin(angle)]
    return pts

def _objective_ring(flat, ring_split=(7, 7)):
    """
    Objective for ring parameter optimization.
    """
    pts = _decode_ring_params(flat, ring_split)
    return _objective_hard(pts.flatten())

def _generate_initial_configs():
    """
    Generate a diverse set of initial configurations.
    """
    configs = []
    rng = np.random.default_rng(42)
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Two concentric rings (7 + 7) with various inner radii
    for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        a1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
        p1 = r_inner * np.column_stack([np.cos(a1), np.sin(a1)])
        p2 = np.column_stack([np.cos(a2), np.sin(a2)])
        configs.append(np.vstack([p1, p2]))
        
    # 3. Three rings (5 + 4 + 5)
    for r1, r2 in [(0.3, 0.6), (0.4, 0.7), (0.5, 0.8)]:
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        a3 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
        p1 = r1 * np.column_stack([np.cos(a1), np.sin(a1)])
        p2 = r2 * np.column_stack([np.cos(a2), np.sin(a2)])
        p3 = np.column_stack([np.cos(a3), np.sin(a3)])
        configs.append(np.vstack([p1, p2, p3]))
    
    # 4. Random configurations in disk
    for seed in range(8):
        rs = np.random.default_rng(seed * 13 + 7)
        r = rs.uniform(0.2, 1.0, N)
        theta = rs.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        configs.append(pts)
        
    # 5. Jittered regular polygon
    base = configs[0]
    for std in [0.05, 0.1, 0.15]:
        jitter = rng.normal(0, std, size=(N, 2))
        configs.append(base + jitter)
        
    # 6. Elliptical configurations
    for rx, ry in [(1.0, 0.7), (1.0, 0.85), (1.2, 0.75)]:
        a = np.linspace(0, 2 * np.pi, N, endpoint=False)
        configs.append(np.column_stack([rx * np.cos(a), ry * np.sin(a)]))
        
    return configs

def _optimize_ring_params(params0, deadline, ring_split=(7, 7)):
    """
    Optimize ring parameters using Nelder-Mead.
    """
    if time.time() > deadline - 5:
        return None
    try:
        res = minimize(_objective_ring, params0, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True,
                                'args': (ring_split,)})
        return res
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_score = 0.0
    best_pts = None
    best_obj = float('inf')
    
    def update_best(pts):
        nonlocal best_score, best_pts, best_obj
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            best_obj = _objective_hard(pts.flatten())

    # Phase 1: Optimize ring parameters (reduced dimensionality)
    # 7+7 ring split
    ring_split = (7, 7)
    n_ring_params = 2 + 7 + 7  # 2 radii + 14 angles
    
    # Initial ring params: r1=0.5, angles evenly spaced, r2=1.0, angles staggered
    r1_init = 0.5
    a1_init = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    r2_init = 1.0
    a2_init = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
    params0 = np.concatenate([[r1_init], a1_init, [r2_init], a2_init])
    
    res_ring = _optimize_ring_params(params0, deadline, ring_split)
    if res_ring is not None and res_ring.fun < best_obj:
        pts_ring = _decode_ring_params(res_ring.x, ring_split)
        update_best(pts_ring)
    
    # Try different initial ring configurations
    for r1_try in [0.3, 0.4, 0.6, 0.7, 0.8]:
        if time.time() > deadline - 30:
            break
        a1_try = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a2_try = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
        params_try = np.concatenate([[r1_try], a1_try, [1.0], a2_try])
        res_try = _optimize_ring_params(params_try, deadline, ring_split)
        if res_try is not None and res_try.fun < best_obj:
            pts_try = _decode_ring_params(res_try.x, ring_split)
            update_best(pts_try)
    
    # Phase 2: Multi-start Nelder-Mead on full 28D space
    inits = _generate_initial_configs()
    for i, pts0 in enumerate(inits):
        if time.time() > deadline - 40:
            break
        try:
            res = minimize(_objective_hard, pts0.flatten(), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            if res.success or res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B with soft objective for smoother gradients
    if best_pts is not None and time.time() < deadline - 25:
        try:
            res_lbfgs = minimize(_objective_soft, best_pts.flatten(), method='L-BFGS-B',
                                 options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res_lbfgs.fun < best_obj:
                update_best(res_lbfgs.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 4: Differential Evolution (Global Search)
    if time.time() < deadline - 15:
        try:
            bounds = [(-1.5, 1.5)] * (N * 2)
            de_res = differential_evolution(
                _objective_hard, bounds,
                seed=42, maxiter=40, tol=1e-8,
                popsize=12, mutation=(0.4, 1.0), recombination=0.7,
                polish=False, update_interval=1,
                init='sobol'
            )
            if de_res.fun < best_obj:
                update_best(de_res.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 5: Final Polish
    if best_pts is not None and time.time() < deadline - 5:
        try:
            # Nelder-Mead with tighter tolerance
            res = minimize(_objective_hard, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
                
            # Powell's Method
            if time.time() < deadline - 2:
                res_powell = minimize(_objective_hard, best_pts.flatten(), method='Powell',
                                      options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-14})
                if res_powell.fun < best_obj:
                    update_best(res_powell.x.reshape(N, 2))
        except Exception:
            pass
    
    # Fallback to regular polygon if nothing worked
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area
    try:
        hull = ConvexHull(best_pts)
        area = hull.volume
        if area > 1e-10:
            scale = np.sqrt(1.0 / area)
            best_pts = best_pts * scale
    except Exception:
        pass
    
    return best_pts
