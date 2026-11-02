"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

DEADLINE = 140.0  # safety margin before 150s kill


def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
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


def _support_function_objective(radii):
    """
    Objective using support function representation.
    radii: array of N radii for fixed angles.
    Points are at angle_i * radius_i.
    This enforces convexity if radii are positive and angles are sorted.
    """
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Check if points form a convex polygon (all cross products same sign)
    # For a convex polygon in CCW order, all cross products should be positive
    valid = True
    for i in range(N):
        p1 = points[i]
        p2 = points[(i + 1) % N]
        p3 = points[(i + 2) % N]
        v1 = p2 - p1
        v2 = p3 - p2
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross < -1e-8:  # Allow small numerical errors
            valid = False
            break
    
    if not valid:
        return 1e10
    
    min_area = _min_triangle_area_vec(points)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area


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
        pts1 = r1 * np.column_stack([np.cos(a1), np.sin(a1)])
        pts2 = r2 * np.column_stack([np.cos(a2), np.sin(a2)])
        pts3 = np.column_stack([np.cos(a3), np.sin(a3)])
        pts = np.vstack([pts1, pts2, pts3])
        inits.append(pts.flatten())

    # 5. Random convex perturbations of regular polygon
    for _ in range(10):
        r = 1.0 + rng.uniform(-0.15, 0.15, N)
        a = angles + rng.uniform(-0.05, 0.05, N)
        pts = np.column_stack([r * np.cos(a), r * np.sin(a)])
        inits.append(pts.flatten())

    return inits


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


def _optimize_support_function(radii0, deadline):
    """Optimize radii in support function representation."""
    if time.time() > deadline - 5:
        return None
    
    try:
        res = minimize(_support_function_objective, radii0, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        if res.success or res.fun < 1e9:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            points = np.column_stack([res.x * np.cos(angles), res.x * np.sin(angles)])
            return points
    except Exception:
        pass
    return None


def construct():
    start_time = time.time()
    
    best_points = None
    best_score = -1.0
    
    # Try multiple starting points
    inits = _make_inits()
    
    for seed_idx, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Try different optimizers
        optimizers = [
            ('nm', _nm, _objective_hard),
            ('lbfgs', _lbfgs, _objective_soft),
            ('powell', _powell, _objective_hard),
        ]
        
        for opt_name, opt_func, obj_func in optimizers:
            if time.time() - start_time > DEADLINE - 10:
                break
            
            try:
                result = opt_func(init, DEADLINE, obj=obj_func)
                if result is not None and (result.success or result.fun < 1e9):
                    opt_points = result.x.reshape(N, 2)
                    s = _score(opt_points)
                    if s > best_score:
                        best_score = s
                        best_points = opt_points.copy()
            except Exception:
                continue
    
    # Try support function representation
    if time.time() - start_time < DEADLINE - 15:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        for r_base in [1.0, 0.9, 1.1]:
            if time.time() - start_time > DEADLINE - 10:
                break
            radii0 = r_base * np.ones(N)
            sf_points = _optimize_support_function(radii0, DEADLINE)
            if sf_points is not None:
                s = _score(sf_points)
                if s > best_score:
                    best_score = s
                    best_points = sf_points.copy()
    
    # If no good result, fall back to regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_points)
    
    # Final refinement: coordinate descent
    if time.time() - start_time < DEADLINE - 5:
        refined = _coordinate_descent(best_points, DEADLINE, max_rounds=30)
        s_refined = _score(refined)
        if s_refined > best_score:
            best_score = s_refined
            best_points = refined.copy()
    
    # Final local search with small perturbations
    if time.time() - start_time < DEADLINE - 2:
        rng = np.random.default_rng(42)
        for trial in range(10):
            if time.time() - start_time > DEADLINE - 2:
                break
            perturbation = rng.randn(N, 2) * 0.02
            trial_points = best_points + perturbation
            s = _score(trial_points)
            if s > best_score:
                best_score = s
                best_points = trial_points.copy()
    
    return best_points
# EVOLVE-BLOCK-END
