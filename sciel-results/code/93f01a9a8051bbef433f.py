"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points. Returns min area (not doubled)."""
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
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-15:
                    return min_area
    return min_area

def _hull_area_fast(points):
    """Compute convex hull area. Returns 0 if degenerate."""
    try:
        hull = ConvexHull(points, qhull_options='Qx')
        return hull.volume
    except Exception:
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

def _score(points):
    """Minimum triangle area divided by convex hull area."""
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _all_triangle_areas(points):
    """Compute all triangle areas as a flat array."""
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
            tri_areas = 0.5 * np.abs(cross)
            if len(tri_areas) > 0:
                areas_list.append(tri_areas)
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

def _objective_hard(points_flat):
    """Minimize hull_area / min_triangle_area."""
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

def _objective_soft(points_flat, beta=200.0):
    """Soft-min objective for smoother optimization with higher beta for better approximation."""
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    # Use a more stable log-sum-exp computation
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        exp_z = np.exp(z - max_z)
        log_sum = max_z + np.log(np.sum(exp_z))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val

def _objective_soft_medium(points_flat, beta=80.0):
    """Medium-temperature soft-min for broader exploration."""
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
        exp_z = np.exp(z - max_z)
        log_sum = max_z + np.log(np.sum(exp_z))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val

def _normalize_unit_area(points):
    """Scale points so convex hull area = 1."""
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

def _nm(x0, deadline, max_iter=3000, obj_func=None):
    """Run Nelder-Mead from x0."""
    if obj_func is None:
        obj_func = _objective_hard
    remaining = deadline - time.time()
    if remaining < 2.0:
        return None
    try:
        res = minimize(
            obj_func,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
        )
        return res.x
    except Exception:
        return None

def _powell(x0, deadline, max_iter=1000, obj_func=None):
    """Run Powell from x0."""
    if obj_func is None:
        obj_func = _objective_hard
    remaining = deadline - time.time()
    if remaining < 3.0:
        return None
    try:
        res = minimize(
            obj_func,
            x0,
            method='Powell',
            options={'maxiter': max_iter, 'xtol': 1e-11, 'ftol': 1e-13}
        )
        return res.x
    except Exception:
        return None

def _lbfgs(x0, deadline, max_iter=500, obj_func=None):
    """Run L-BFGS-B from x0."""
    if obj_func is None:
        obj_func = _objective_soft
    remaining = deadline - time.time()
    if remaining < 3.0:
        return None
    try:
        res = minimize(
            obj_func,
            x0,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-13, 'gtol': 1e-10}
        )
        return res.x
    except Exception:
        return None

def _coordinate_descent(pts, deadline, max_rounds=60):
    """Cyclic coordinate descent on the hard score with adaptive step sizing."""
    pts = pts.copy()
    n = len(pts)
    step0 = 0.03
    for rnd in range(max_rounds):
        if time.time() > deadline - 10:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline - 10:
                break
            for coord in range(2):
                orig = pts[i, coord]
                step = step0
                best_val = _score(pts)
                best_pos = orig
                for _ in range(25):
                    pts[i, coord] = orig + step
                    s_plus = _score(pts)
                    pts[i, coord] = orig - step
                    s_minus = _score(pts)
                    pts[i, coord] = orig
                    
                    if s_plus > best_val:
                        best_val = s_plus
                        best_pos = orig + step
                    elif s_minus > best_val:
                        best_val = s_minus
                        best_pos = orig - step
                    
                    if s_plus > s_minus and s_plus > best_val - 1e-15:
                        pts[i, coord] = orig + step
                    elif s_minus > s_plus and s_minus > best_val - 1e-15:
                        pts[i, coord] = orig - step
                    else:
                        break
                    improved = True
                    step *= 0.4
                    if step < 1e-11:
                        break
                if best_pos != orig:
                    pts[i, coord] = best_pos
        if not improved:
            break
    return pts

def _final_polish(pts_flat, deadline):
    """High-precision final polish using multiple optimization methods with tight tolerances."""
    pts = pts_flat.reshape(N, 2).copy()
    best_obj = _objective_hard(pts_flat)
    best_pts = pts.copy()
    
    # L-BFGS-B with very tight tolerance and higher beta for sharper min
    if time.time() < deadline - 12:
        try:
            res = minimize(
                lambda x: _objective_soft(x, beta=300.0),
                pts_flat,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 150000}
            )
            obj_val = _objective_hard(res.x)
            if obj_val < best_obj:
                best_obj = obj_val
                best_pts = res.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    # Powell with tight tolerance
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _objective_hard,
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 8000, 'xtol': 1e-14, 'ftol': 1e-16}
            )
            obj_val = _objective_hard(res.x)
            if obj_val < best_obj:
                best_obj = obj_val
                best_pts = res.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    # Coordinate descent with many rounds
    if time.time() < deadline - 8:
        pts_cd = _coordinate_descent(best_pts, deadline, max_rounds=80)
        obj_val = _objective_hard(pts_cd.flatten())
        if obj_val < best_obj:
            best_obj = obj_val
            best_pts = pts_cd.copy()
    
    # One more L-BFGS-B pass after CD
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _objective_soft(x, beta=400.0),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 80000}
            )
            obj_val = _objective_hard(res.x)
            if obj_val < best_obj:
                best_obj = obj_val
                best_pts = res.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    return best_pts.flatten()

def _hull_pull(points, alpha=0.2):
    """Expand convex hull outward with adaptive alpha based on current geometry."""
    try:
        hull = ConvexHull(points)
        hull_idx = set(hull.vertices.tolist())
    except Exception:
        return points.copy()
    centroid = points.mean(axis=0)
    new_pts = points.copy()
    
    # Compute average radius for adaptive scaling
    radii = np.linalg.norm(points[hull_idx] - centroid, axis=1)
    avg_radius = np.mean(radii)
    
    for idx in hull_idx:
        v = points[idx] - centroid
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            # Scale alpha inversely with radius to avoid over-expanding
            local_alpha = alpha * (avg_radius / (norm + 1e-12))
            new_pts[idx] = points[idx] + local_alpha * v
    return new_pts

def _make_inits():
    """Generate diverse initializations."""
    inits = []
    rng = np.random.default_rng(42)
    
    # Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_points = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(reg_points.flatten())
    
    # Rotated regular polygons
    for rot in [0.1, 0.3, 0.7, 1.0, 1.5, 2.0]:
        a = angles + rot
        pts = np.column_stack([np.cos(a), np.sin(a)])
        inits.append(pts.flatten())
    
    # Jittered regular polygon
    for jitter_scale in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        jitter = rng.normal(0, jitter_scale, size=(N, 2))
        pts = reg_points + jitter
        inits.append(pts.flatten())
    
    # Random points on circle with varying radii
    for seed in range(8):
        rng_s = np.random.default_rng(seed * 1000 + 7)
        angles_r = np.sort(rng_s.uniform(0, 2 * np.pi, N))
        radii = rng_s.uniform(0.7, 1.3, N)
        pts = np.column_stack([radii * np.cos(angles_r), radii * np.sin(angles_r)])
        inits.append(pts.flatten())
    
    # Two concentric rings
    for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for n_inner in [5, 6, 7]:
            n_outer = N - n_inner
            a_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
            a_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False) + np.pi/n_outer
            pts_inner = np.column_stack([r_inner * np.cos(a_inner), r_inner * np.sin(a_inner)])
            pts_outer = np.column_stack([np.cos(a_outer), np.sin(a_outer)])
            pts = np.vstack([pts_inner, pts_outer])
            inits.append(pts.flatten())
    
    # Elliptical distributions
    for ax, ay in [(1.0, 0.5), (1.0, 0.7), (1.2, 0.8), (1.3, 0.9), (1.5, 0.7), (1.8, 0.6)]:
        angles_e = np.linspace(0, 2*np.pi, N, endpoint=False)
        pts = np.column_stack([ax * np.cos(angles_e), ay * np.sin(angles_e)])
        inits.append(pts.flatten())
    
    # Random convex-like: sort random points by angle from centroid
    for seed in range(8):
        rng_s = np.random.default_rng(seed * 777 + 13)
        pts_raw = rng_s.uniform(-1, 1, size=(N, 2))
        centroid = pts_raw.mean(axis=0)
        angles_c = np.arctan2(pts_raw[:,1] - centroid[1], pts_raw[:,0] - centroid[0])
        sorted_idx = np.argsort(angles_c)
        pts = pts_raw[sorted_idx]
        inits.append(pts.flatten())
    
    # Distorted square
    for distort in [0.0, 0.1, 0.2, 0.3, 0.4]:
        t = np.linspace(0, 1, 3, endpoint=False)
        bottom = np.column_stack([t + distort*(1-t)*t, np.zeros(3)])
        top = np.column_stack([1-t - distort*(1-t)*t, np.ones(3)])
        left = np.column_stack([np.zeros(3), t + distort*(1-t)*t])
        right = np.column_stack([np.ones(3), 1-t - distort*(1-t)*t])
        center = np.array([[0.5, 0.5], [0.5 + 0.1*distort, 0.5 - 0.1*distort]])
        pts = np.vstack([bottom, top, left, right, center])
        pts = pts - 0.5
        inits.append(pts.flatten())
    
    # Fibonacci spiral
    golden_angle = 2.399963229728653
    for r_scale in [1.0, 0.8, 1.2, 1.5]:
        pts_fib = np.zeros((N, 2))
        for i in range(N):
            angle = i * golden_angle
            r = r_scale * np.sqrt((i + 0.5) / N)
            pts_fib[i, 0] = r * np.cos(angle)
            pts_fib[i, 1] = r * np.sin(angle)
        inits.append(pts_fib.flatten())
    
    # Regular polygon with alternating radii
    for alt in [0.1, 0.2, 0.3, 0.4]:
        angles_alt = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii_alt = np.ones(N)
        radii_alt[::2] = 1.0 + alt
        radii_alt[1::2] = 1.0 - alt
        pts_alt = np.column_stack([radii_alt * np.cos(angles_alt), radii_alt * np.sin(angles_alt)])
        inits.append(pts_alt.flatten())
    
    # More random starts
    for seed in range(15):
        rng_s = np.random.default_rng(seed * 12345 + 999)
        pts_rand = rng_s.uniform(-1, 1, size=(N, 2))
        norms = np.linalg.norm(pts_rand, axis=1)
        mask = norms > 1
        pts_rand[mask] = pts_rand[mask] / norms[mask, None]
        angles_rand = np.arctan2(pts_rand[:,1], pts_rand[:,0])
        sorted_idx = np.argsort(angles_rand)
        pts_rand = pts_rand[sorted_idx]
        inits.append(pts_rand.flatten())
    
    # Hexagonal lattice-like
    for scale in [0.5, 0.7, 0.9]:
        pts_hex = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                x = scale * (i + 0.5 * (j % 2))
                y = scale * j * np.sqrt(3) / 2
                pts_hex.append([x, y])
        pts_hex = np.array(pts_hex[:N])
        inits.append(pts_hex.flatten())
    
    # Interleaved circles (from inspiration)
    for r1, r2 in [(0.4, 0.8), (0.5, 0.9), (0.6, 1.0)]:
        a1 = np.linspace(0, 2*np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
        pts1 = r1 * np.column_stack([np.cos(a1), np.sin(a1)])
        pts2 = r2 * np.column_stack([np.cos(a2), np.sin(a2)])
        pts = np.vstack([pts1, pts2])
        inits.append(pts.flatten())
    
    return inits

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    inits = _make_inits()
    max_inits = min(len(inits), 60)
    inits = inits[:max_inits]
    
    best_x = None
    best_obj = float('inf')
    
    # Phase 1: NM from each diverse start
    for idx, x0 in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        time_per_init = min(6.0, max(2.0, remaining / max(1, len(inits) - idx)))
        
        res = _nm(x0, deadline, max_iter=int(time_per_init * 800))
        if res is not None:
            obj_val = _objective_hard(res)
            if obj_val < best_obj:
                best_obj = obj_val
                best_x = res.copy()
        
        # Also try soft-NM
        if time.time() < deadline - 8:
            res_soft = _nm(x0, deadline - 8, max_iter=400, obj_func=_objective_soft)
            if res_soft is not None:
                obj_val = _objective_hard(res_soft)
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_x = res_soft.copy()
    
    # Phase 2: Powell polish
    if best_x is not None and time.time() < deadline - 18:
        res_powell = _powell(best_x, deadline, max_iter=3000)
        if res_powell is not None:
            obj_val = _objective_hard(res_powell)
            if obj_val < best_obj:
                best_obj = obj_val
                best_x = res_powell.copy()
    
    # Phase 3: L-BFGS-B with soft objective (finer convergence on smooth landscape)
    if best_x is not None and time.time() < deadline - 15:
        res_lbfgs = _lbfgs(best_x, deadline, max_iter=2000)
        if res_lbfgs is not None:
            obj_val = _objective_hard(res_lbfgs)
            if obj_val < best_obj:
                best_obj = obj_val
                best_x = res_lbfgs.copy()
        
        # Also try medium-temperature soft objective for broader exploration
        if time.time() < deadline - 12:
            try:
                res_medium = minimize(
                    _objective_soft_medium,
                    best_x,
                    method='L-BFGS-B',
                    options={'maxiter': 1200, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                obj_val = _objective_hard(res_medium.x)
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_x = res_medium.x.copy()
            except Exception:
                pass
    
    # Phase 4: Coordinate descent refinement
    if best_x is not None and time.time() < deadline - 12:
        pts_cd = best_x.reshape(N, 2)
        pts_cd = _coordinate_descent(pts_cd, deadline, max_rounds=60)
        obj_val = _objective_hard(pts_cd.flatten())
        if obj_val < best_obj:
            best_obj = obj_val
            best_x = pts_cd.flatten()
    
    # Phase 5: Hull pull + re-optimize (structural move to enlarge hull)
    if best_x is not None and time.time() < deadline - 15:
        pts_pull = _hull_pull(best_x.reshape(N, 2), alpha=0.25)
        obj_pull = _objective_hard(pts_pull.flatten())
        if obj_pull < best_obj:
            best_obj = obj_pull
            best_x = pts_pull.flatten()
            # Re-optimize after hull pull
            if time.time() < deadline - 10:
                res_pull = _nm(best_x, deadline, max_iter=2500)
                if res_pull is not None:
                    obj_val = _objective_hard(res_pull)
                    if obj_val < best_obj:
                        best_obj = obj_val
                        best_x = res_pull.copy()
    
    # Phase 6: Another Powell
    if best_x is not None and time.time() < deadline - 10:
        res_powell2 = _powell(best_x, deadline, max_iter=2000)
        if res_powell2 is not None:
            obj_val = _objective_hard(res_powell2)
            if obj_val < best_obj:
                best_obj = obj_val
                best_x = res_powell2.copy()
    
    # Phase 7: High-precision final polish
    if best_x is not None and time.time() < deadline - 10:
        best_x = _final_polish(best_x, deadline)
        best_obj = _objective_hard(best_x)
    
    # Fallback
    if best_x is None:
        best_x = inits[0].copy()
    
    final_points = best_x.reshape(N, 2)
    final_points = _normalize_unit_area(final_points)
    
    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            reg_points = np.column_stack([np.cos(angles), np.sin(angles)])
            final_points = _normalize_unit_area(reg_points)
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        reg_points = np.column_stack([np.cos(angles), np.sin(angles)])
        final_points = _normalize_unit_area(reg_points)
    
    # Final safety: ensure no NaN/Inf
    if np.any(np.isnan(final_points)) or np.any(np.isinf(final_points)):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        reg_points = np.column_stack([np.cos(angles), np.sin(angles)])
        final_points = _normalize_unit_area(reg_points)
    
    return final_points
# EVOLVE-BLOCK-END
