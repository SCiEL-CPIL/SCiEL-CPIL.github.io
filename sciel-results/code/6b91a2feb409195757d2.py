"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points (returns min area, not doubled)."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area_fast(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

def _objective_hard(points_flat):
    """Minimize hull_area / min_triangle_area (scale-invariant)."""
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    min_area = _min_triangle_area_vec(points)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area

def _objective_soft(points_flat, beta=150.0):
    """Smooth soft-min objective: hull_area / soft_min(triangle areas)."""
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val

def _normalize_unit_area(points):
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            return points * scale
    except Exception:
        pass
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        scale = 1.0 / np.sqrt(w * h)
        return points * scale
    return points

def _hull_pull(points, alpha=0.5):
    """Expand convex hull outward (structural move to enlarge hull area, improving ratio).
    Points on hull are pushed radially outward from centroid by alpha*(radius)."""
    try:
        hull = ConvexHull(points)
        hull_idx = set(hull.vertices.tolist())
    except Exception:
        return points.copy()
    centroid = points.mean(axis=0)
    new_pts = points.copy()
    for idx in hull_idx:
        v = points[idx] - centroid
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            new_pts[idx] = points[idx] + alpha * v
    return new_pts

def _best_of_two(a, b):
    sa = _score(a)
    sb = _score(b)
    return (a, sa) if sa >= sb else (b, sb)

def _nm(x0, deadline, max_iter=2500, obj=_objective_hard):
    remaining = deadline - time.time()
    if remaining < 2.0:
        return None
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        return res
    except Exception:
        return None

def _lbfgs(x0, deadline, max_iter=5000, obj=_objective_soft):
    remaining = deadline - time.time()
    if remaining < 5.0:
        return None
    try:
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        return res
    except Exception:
        return None

def _powell(x0, deadline, max_iter=1500, obj=_objective_hard):
    remaining = deadline - time.time()
    if remaining < 3.0:
        return None
    try:
        res = minimize(obj, x0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-11, 'ftol': 1e-13})
        return res
    except Exception:
        return None

def _coordinate_descent(pts, deadline, max_rounds=50):
    """Cyclic coordinate descent on the hard objective."""
    pts = pts.copy()
    n = len(pts)
    step0 = 0.05
    for rnd in range(max_rounds):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            for coord in range(2):
                orig = pts[i, coord]
                step = step0
                for _ in range(30):
                    pts[i, coord] = orig + step
                    s_plus = _score(pts)
                    pts[i, coord] = orig - step
                    s_minus = _score(pts)
                    pts[i, coord] = orig
                    if s_plus > s_minus:
                        pts[i, coord] = orig + step
                    elif s_minus > s_plus:
                        pts[i, coord] = orig - step
                    else:
                        break
                    improved = True
                    step *= 0.5
                    if step < 1e-8:
                        break
        if not improved:
            break
    return pts

# ---------- Initializations ----------

def _make_inits():
    inits = []
    rng = np.random.default_rng(2024)

    # 1. Regular 14-gon (convex, all on circle)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(reg.flatten())

    # 2. Rotated regular polygons
    for rot in [0.1, 0.4, 0.9, 1.3, 2.0, 3.5]:
        a = angles + rot
        inits.append(np.column_stack([np.cos(a), np.sin(a)]).flatten())

    # 3. Two concentric rings (7 inner, 7 outer) - key structural family
    for r_inner in [0.3, 0.45, 0.6, 0.75, 0.85]:
        a_in = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a_out = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
        pts_in = r_inner * np.column_stack([np.cos(a_in), np.sin(a_in)])
        pts_out = np.column_stack([np.cos(a_out), np.sin(a_out)])
        pts = np.vstack([pts_in, pts_out])
        inits.append(pts.flatten())

    # 4. 3 rings (5, 5, 4)
    for r1, r2 in [(0.3, 0.65), (0.4, 0.8), (0.5, 0.9)]:
        a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
        a3 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        p1 = r1 * np.column_stack([np.cos(a1), np.sin(a1)])
        p2 = r2 * np.column_stack([np.cos(a2), np.sin(a2)])
        p3 = np.column_stack([np.cos(a3), np.sin(a3)])
        pts = np.vstack([p1, p2, p3])
        inits.append(pts.flatten())

    # 5. Random perturbations of regular polygon
    for seed in [1, 2, 3]:
        rng_s = np.random.RandomState(seed)
        angles_p = angles + rng_s.normal(0, 0.05, N)
        radii = 1.0 + rng_s.normal(0, 0.1, N)
        radii = np.clip(radii, 0.5, 1.5)
        pts = np.column_stack([radii * np.cos(angles_p), radii * np.sin(angles_p)])
        inits.append(pts.flatten())

    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    inits = _make_inits()
    
    best_points = None
    best_score = 0.0
    
    # Phase 1: Evaluate all initializations and pick the best few
    init_scores = []
    for init_flat in inits:
        if time.time() > deadline - 30:
            break
        pts = init_flat.reshape(N, 2)
        s = _score(pts)
        init_scores.append((s, pts))
    
    init_scores.sort(key=lambda x: -x[0])
    
    # Take top 3 initializations for refinement
    top_inits = [x[1] for x in init_scores[:3]]
    if not top_inits:
        top_inits = [inits[0].reshape(N, 2)]
    
    for init_pts in top_inits:
        if time.time() > deadline - 20:
            break
        
        # Phase 2: L-BFGS with soft objective (smooth, gradient-based)
        if time.time() < deadline - 15:
            res = _lbfgs(init_pts.flatten(), deadline, max_iter=3000, obj=_objective_soft)
            if res is not None and res.fun < 1e9:
                cand_pts = res.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand_pts.copy()
        
        # Phase 3: Nelder-Mead with hard objective (non-smooth, robust)
        if time.time() < deadline - 10:
            current = best_points if best_points is not None else init_pts
            res = _nm(current.flatten(), deadline, max_iter=2000, obj=_objective_hard)
            if res is not None and res.fun < 1e9:
                cand_pts = res.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand_pts.copy()
        
        # Phase 4: Powell refinement
        if time.time() < deadline - 8:
            current = best_points if best_points is not None else init_pts
            res = _powell(current.flatten(), deadline, max_iter=1000, obj=_objective_hard)
            if res is not None and res.fun < 1e9:
                cand_pts = res.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand_pts.copy()
        
        # Phase 5: Coordinate descent for fine-tuning
        if time.time() < deadline - 5:
            current = best_points if best_points is not None else init_pts
            cand_pts = _coordinate_descent(current, deadline, max_rounds=30)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand_pts.copy()
    
    # Final safety check
    if best_points is None or _hull_area_fast(best_points) < 1e-10:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
