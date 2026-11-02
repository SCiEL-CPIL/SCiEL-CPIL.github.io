"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
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

# Fixed non-uniform angles: cluster near diagonals (45°, 135°, 225°, 315°) to resolve small triangles
_FIXED_ANGLES = np.array([
    0.0,
    np.pi/8 * 0.7,    # ~15.75°
    np.pi/8 * 1.3,    # ~29.25°
    np.pi/4 * 0.85,   # ~38.25°
    np.pi/4 * 1.15,   # ~51.75°
    np.pi/4 + np.pi/8 * 0.7,   # ~69.75°
    np.pi/4 + np.pi/8 * 1.3,   # ~83.25°
    np.pi/2 * 0.85,   # ~101.25°
    np.pi/2 * 1.15,   # ~114.75°
    3*np.pi/4 * 0.85, # ~132.75°
    3*np.pi/4 * 1.15, # ~146.25°
    3*np.pi/4 + np.pi/8 * 0.7,  # ~164.25°
    3*np.pi/4 + np.pi/8 * 1.3,  # ~177.75°
    np.pi,
    5*np.pi/4 - np.pi/8 * 1.3,  # ~192.25°
    5*np.pi/4 - np.pi/8 * 0.7,  # ~205.75°
    5*np.pi/4 * 0.85, # ~223.5°
    5*np.pi/4 * 1.15, # ~236.6°
    3*np.pi/2 - np.pi/8 * 0.7,  # ~255.75°
    3*np.pi/2 - np.pi/8 * 1.3,  # ~269.25°
    7*np.pi/4 * 0.85, # ~287.25°
    7*np.pi/4 * 1.15, # ~300.75°
    7*np.pi/4 + np.pi/8 * 0.7,  # ~318.75°
    7*np.pi/4 + np.pi/8 * 1.3,  # ~332.25°
    2*np.pi - np.pi/8 * 1.3,    # ~345.75°
    2*np.pi - np.pi/8 * 0.7,    # ~359.25°
])
# Select 14 angles from a denser set, clustered near diagonals
def _make_angles(n=N):
    """Create n angles clustered near the 4 diagonal directions."""
    base = np.linspace(0, 2*np.pi, n, endpoint=False)
    # Perturb toward diagonals: pull each angle toward nearest diagonal
    diags = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
    angles = np.zeros(n)
    for i in range(n):
        # Find nearest diagonal
        diffs = np.abs(base[i] - diags)
        diffs = np.minimum(diffs, 2*np.pi - diffs)
        nearest = diags[np.argmin(diffs)]
        # Blend: 30% toward diagonal
        angles[i] = 0.7 * base[i] + 0.3 * nearest
        # Handle wrap-around
        if angles[i] > 2*np.pi:
            angles[i] -= 2*np.pi
    angles.sort()
    return angles

_ANGLES = _make_angles(N)

def _radii_to_points(radii):
    """Convert radii at fixed angles to point coordinates."""
    radii = np.maximum(radii, 0.01)  # Ensure positive
    pts = np.column_stack([radii * np.cos(_ANGLES), radii * np.sin(_ANGLES)])
    return pts

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

def _objective_radii(radii_flat):
    """Optimize radii directly. Minimize hull_area / min_triangle_area."""
    if np.any(np.isnan(radii_flat)) or np.any(np.isinf(radii_flat)):
        return 1e10
    radii = np.maximum(radii_flat, 0.01)
    pts = _radii_to_points(radii)
    min_area = _min_triangle_area_vec(pts)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area

def _objective_radii_soft(radii_flat, beta=200.0):
    """Soft-min objective on radii for smoother optimization."""
    if np.any(np.isnan(radii_flat)) or np.any(np.isinf(radii_flat)):
        return 1e10
    radii = np.maximum(radii_flat, 0.01)
    pts = _radii_to_points(radii)
    areas = _all_triangle_areas(pts)
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
    hull_area = _hull_area_fast(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val

def _make_radii_inits():
    """Generate diverse initializations in radii space."""
    inits = []
    rng = np.random.default_rng(42)
    
    # Uniform radii (regular polygon)
    inits.append(np.ones(N))
    
    # Varying base scales
    for scale in [0.8, 1.0, 1.2, 1.5]:
        inits.append(np.ones(N) * scale)
    
    # Elliptical radii
    for ax, ay in [(1.0, 0.5), (1.0, 0.7), (1.2, 0.8), (1.3, 0.9), (1.5, 0.7), (1.8, 0.6), (0.7, 1.3)]:
        radii = ax * np.cos(_ANGLES) + ay * np.sin(_ANGLES)
        radii = np.abs(radii)
        radii = np.maximum(radii, 0.1)
        inits.append(radii)
    
    # Alternating radii
    for alt in [0.1, 0.2, 0.3, 0.4, 0.5]:
        radii = np.ones(N)
        radii[::2] = 1.0 + alt
        radii[1::2] = 1.0 - alt
        inits.append(radii)
    
    # Random radii with positive constraint
    for seed in range(20):
        rng_s = np.random.default_rng(seed * 12345 + 999)
        radii = rng_s.uniform(0.5, 1.5, N)
        inits.append(radii)
    
    # Smooth random variations
    for seed in range(15):
        rng_s = np.random.default_rng(seed * 777 + 13)
        # Low-frequency smooth variation
        freq = rng_s.uniform(1, 4, 2)
        phase = rng_s.uniform(0, 2*np.pi, 2)
        amp = rng_s.uniform(0.1, 0.4, 2)
        radii = 1.0 + amp[0]*np.sin(freq[0]*_ANGLES + phase[0]) + amp[1]*np.cos(freq[1]*_ANGLES + phase[1])
        radii = np.maximum(radii, 0.1)
        inits.append(radii)
    
    # Diagonal-biased: larger radii near diagonals
    for bias in [0.1, 0.2, 0.3, 0.5]:
        diag_mask = np.zeros(N)
        for d in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
            dist = np.minimum(np.abs(_ANGLES - d), 2*np.pi - np.abs(_ANGLES - d))
            diag_mask += np.exp(-dist**2 / 0.3)
        radii = 1.0 + bias * diag_mask
        inits.append(radii)
    
    # Concentric ring patterns
    for r_inner in [0.3, 0.4, 0.5, 0.6]:
        # Alternate between inner and outer radii
        radii = np.ones(N)
        radii[::2] = r_inner
        radii[1::2] = 1.0
        inits.append(radii)
    
    # Fibonacci-like radii
    golden_angle = 2.399963229728653
    for r_scale in [1.0, 0.8, 1.2]:
        radii = np.ones(N)
        for i in range(N):
            radii[i] = r_scale * (0.7 + 0.3 * np.sin(i * golden_angle))
        radii = np.maximum(radii, 0.1)
        inits.append(radii)
    
    # Random with structure: sort by angle, create smooth profile
    for seed in range(10):
        rng_s = np.random.default_rng(seed * 31415 + 2718)
        # Create a smooth random profile
        t = np.linspace(0, 2*np.pi, N, endpoint=False)
        coeffs = rng_s.uniform(-0.3, 0.3, 4)
        radii = 1.0 + coeffs[0]*np.sin(t) + coeffs[1]*np.cos(t) + coeffs[2]*np.sin(2*t) + coeffs[3]*np.cos(2*t)
        radii = np.maximum(radii, 0.1)
        inits.append(radii)
    
    return inits

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    Uses support-function (radii) representation at fixed non-uniform angles.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_radii = None
    best_obj = float('inf')
    
    # Phase 1: Multi-start Nelder-Mead in radii space
    radii_inits = _make_radii_inits()
    max_inits = min(len(radii_inits), 50)
    radii_inits = radii_inits[:max_inits]
    
    for idx, r0 in enumerate(radii_inits):
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        time_per_init = min(5.0, max(1.5, remaining / max(1, len(radii_inits) - idx)))
        
        # Hard objective NM
        try:
            res = minimize(
                _objective_radii,
                r0,
                method='Nelder-Mead',
                options={'maxiter': int(time_per_init * 1000), 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            obj_val = _objective_radii(res.x)
            if obj_val < best_obj:
                best_obj = obj_val
                best_radii = res.x.copy()
        except Exception:
            pass
        
        # Soft objective NM for smoother landscape
        if time.time() < deadline - 8:
            try:
                res_soft = minimize(
                    lambda r: _objective_radii_soft(r, beta=150.0),
                    r0,
                    method='Nelder-Mead',
                    options={'maxiter': int(time_per_init * 500), 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                obj_val = _objective_radii(res_soft.x)
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_radii = res_soft.x.copy()
            except Exception:
                pass
    
    # Phase 2: Powell polish in radii space
    if best_radii is not None and time.time() < deadline - 15:
        try:
            res_powell = minimize(
                _objective_radii,
                best_radii,
                method='Powell',
                options={'maxiter': 5000, 'xtol': 1e-12, 'ftol': 1e-14}
            )
            obj_val = _objective_radii(res_powell.x)
            if obj_val < best_obj:
                best_obj = obj_val
                best_radii = res_powell.x.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B with soft objective in radii space
    if best_radii is not None and time.time() < deadline - 12:
        try:
            res_lbfgs = minimize(
                lambda r: _objective_radii_soft(r, beta=250.0),
                best_radii,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 100000}
            )
            obj_val = _objective_radii(res_lbfgs.x)
            if obj_val < best_obj:
                best_obj = obj_val
                best_radii = res_lbfgs.x.copy()
        except Exception:
            pass
    
    # Phase 4: Convert to points and do coordinate descent
    if best_radii is not None and time.time() < deadline - 12:
        pts = _radii_to_points(best_radii)
        pts_cd = _coordinate_descent(pts, deadline, max_rounds=50)
        # Try to fit radii back from optimized points
        # Project points onto our fixed angles
        new_radii = np.zeros(N)
        for i in range(N):
            # Find best radius for this angle
            best_r = 0.5
            for r_test in np.linspace(0.1, 2.0, 20):
                test_pts = pts.copy()
                test_pts[i] = r_test * np.array([np.cos(_ANGLES[i]), np.sin(_ANGLES[i])])
                s = _score(test_pts)
                if s > _score(pts):
                    best_r = r_test
            new_radii[i] = best_r
        obj_new = _objective_radii(new_radii)
        if obj_new < best_obj:
            best_obj = obj_new
            best_radii = new_radii.copy()
        else:
            # Keep the point-space result if better
            obj_pts = _score(pts_cd)
            if obj_pts > _score(pts):
                # Use point-space result directly
                best_points_from_cd = pts_cd
                best_obj = _hull_area_fast(pts_cd) / _min_triangle_area_vec(pts_cd)
    
    # Phase 5: Differential evolution for broader search
    if time.time() < deadline - 20:
        try:
            bounds = [(0.2, 2.0)] * N
            de_res = differential_evolution(
                _objective_radii,
                bounds,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                seed=42,
                polish=False,
                updating='deferred'
            )
            obj_de = _objective_radii(de_res.x)
            if obj_de < best_obj:
                best_obj = obj_de
                best_radii = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 6: Final high-precision polish
    if best_radii is not None and time.time() < deadline - 10:
        # L-BFGS-B with very tight tolerance
        try:
            res_final = minimize(
                lambda r: _objective_radii_soft(r, beta=400.0),
                best_radii,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxfun': 200000}
            )
            obj_val = _objective_radii(res_final.x)
            if obj_val < best_obj:
                best_obj = obj_val
                best_radii = res_final.x.copy()
        except Exception:
            pass
        
        # Powell with tight tolerance
        if time.time() < deadline - 8:
            try:
                res_powell2 = minimize(
                    _objective_radii,
                    best_radii,
                    method='Powell',
                    options={'maxiter': 10000, 'xtol': 1e-14, 'ftol': 1e-16}
                )
                obj_val = _objective_radii(res_powell2.x)
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_radii = res_powell2.x.copy()
            except Exception:
                pass
    
    # Fallback
    if best_radii is None:
        best_radii = np.ones(N)
    
    final_points = _radii_to_points(best_radii)
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
