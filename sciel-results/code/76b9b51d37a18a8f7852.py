"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations
import sys

# ---------- Core geometry ----------

def _min_triangle_area_vec(points):
    """
    Highly optimized minimum triangle area calculation.
    Uses vectorized operations for each pair (i, j) to compute areas with all k > j.
    """
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    # We only need to check i < j < k
    for i in range(n - 2):
        xi, yi = x[i], y[i]
        # Vectorized differences from i to j+1..n-1
        x_ji = x[i+1:] - xi
        y_ji = y[i+1:] - yi
        
        # For each j in i+1..n-2, we need cross products with k > j
        # This is slightly tricky to fully vectorize across j and k simultaneously 
        # without a large temporary array, but for N=14 it's fine.
        # Let's iterate j, but vectorize k.
        for j in range(i + 1, n - 1):
            # Vector for all k > j
            # x_k = x[j+1:], y_k = y[j+1:]
            # v1 = (x_j - x_i, y_j - y_i)
            # v2_k = (x_k - x_i, y_k - y_i)
            # cross = v1_x * v2_y - v1_y * v2_x
            
            v1_x = x[j] - xi
            v1_y = y[j] - yi
            
            v2_x = x[j+1:] - xi
            v2_y = y[j+1:] - yi
            
            cross = v1_x * v2_y - v1_y * v2_x
            areas = 0.5 * np.abs(cross)
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                # Early exit if area is effectively zero
                if min_area < 1e-14:
                    return min_area
    return min_area

def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        xi, yi = x[i], y[i]
        x_ji = x[i+1:] - xi
        y_ji = y[i+1:] - yi
        for j in range(i + 1, n - 1):
            v1_x = x[j] - xi
            v1_y = y[j] - yi
            v2_x = x[j+1:] - xi
            v2_y = y[j+1:] - yi
            cross = v1_x * v2_y - v1_y * v2_x
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

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

# ---------- Direct 28-parameter optimization ----------

def _neg_score_direct(points_flat):
    """
    Negative score for direct 28-parameter optimization.
    points_flat: 28-element array of 14 (x,y) coordinates.
    """
    if np.any(~np.isfinite(points_flat)):
        return 1e10
    points = points_flat.reshape(14, 2)
    
    # Check for degenerate cases
    if np.any(np.abs(points) > 1e6):
        return 1e10
    
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    
    min_area = _min_triangle_area_vec(points)
    if min_area < 1e-12:
        return 1e10
    
    return -(min_area / hull_area)

def _soft_min_value(areas, beta):
    """Compute soft minimum of areas using log-sum-exp with adaptive stability."""
    if len(areas) == 0:
        return 0.0
    a_min = np.min(areas)
    if a_min < 1e-12:
        return a_min
    z = -beta * (areas - a_min)
    log_sum = np.log(np.sum(np.exp(z)))
    return a_min + log_sum / beta

def _neg_score_soft_direct(points_flat, beta=100.0):
    """
    Soft negative score for direct 28-parameter optimization using log-sum-exp.
    """
    if np.any(~np.isfinite(points_flat)):
        return 1e10
    points = points_flat.reshape(14, 2)
    
    if np.any(np.abs(points) > 1e6):
        return 1e10
    
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    
    soft_min_val = _soft_min_value(areas, beta)
    if soft_min_val < 1e-12:
        return 1e10
        
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    
    return -(soft_min_val / hull_area)

def _generate_direct_inits():
    """Generate diverse initial 28-point configurations for direct optimization."""
    inits = []
    
    # Regular 14-gon
    angles = np.linspace(0, 2*np.pi, 14, endpoint=False)
    regular = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(regular.flatten())
    
    # Rotated regular 14-gons
    for rot in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        a = angles + rot
        pts = np.column_stack([np.cos(a), np.sin(a)])
        inits.append(pts.flatten())
    
    # Elliptical configurations
    for a_ratio in [0.5, 0.7, 0.85, 1.2, 1.5, 2.0]:
        a = angles
        pts = np.column_stack([np.cos(a) * a_ratio, np.sin(a)])
        inits.append(pts.flatten())
    
    # Hexagonal lattice-inspired: 3 rows of 5, 5, 4 points
    for seed in range(10):
        rng = np.random.default_rng(seed * 7 + 3)
        pts = np.zeros((14, 2))
        # Row 1: 5 points
        x1 = np.linspace(-1.5, 1.5, 5) + rng.normal(0, 0.1, 5)
        y1 = 0.8 + rng.normal(0, 0.1, 5)
        pts[:5, 0] = x1
        pts[:5, 1] = y1
        # Row 2: 5 points
        x2 = np.linspace(-1.5, 1.5, 5) + rng.normal(0, 0.1, 5)
        y2 = 0.0 + rng.normal(0, 0.1, 5)
        pts[5:10, 0] = x2
        pts[5:10, 1] = y2
        # Row 3: 4 points
        x3 = np.linspace(-1.2, 1.2, 4) + rng.normal(0, 0.1, 4)
        y3 = -0.8 + rng.normal(0, 0.1, 4)
        pts[10:14, 0] = x3
        pts[10:14, 1] = y3
        inits.append(pts.flatten())
    
    # Radial with varying radii
    for seed in range(15):
        rng = np.random.default_rng(seed * 13 + 7)
        a = np.sort(rng.uniform(0, 2*np.pi, 14))
        r = 1.0 + rng.normal(0, 0.3, 14)
        r = np.maximum(r, 0.5)
        pts = np.column_stack([r * np.cos(a), r * np.sin(a)])
        inits.append(pts.flatten())
    
    # Two concentric heptagons
    for r_inner in [0.4, 0.5, 0.6, 0.7]:
        for rot in [0.0, np.pi/7, np.pi/14]:
            a_outer = np.linspace(0, 2*np.pi, 7, endpoint=False) + rot
            a_inner = np.linspace(0, 2*np.pi, 7, endpoint=False)
            r_outer = np.ones(7)
            r_inner_arr = np.full(7, r_inner)
            pts_outer = np.column_stack([r_outer * np.cos(a_outer), r_outer * np.sin(a_outer)])
            pts_inner = np.column_stack([r_inner_arr * np.cos(a_inner), r_inner_arr * np.sin(a_inner)])
            pts = np.vstack([pts_outer, pts_inner])
            # Shuffle to avoid ordering bias
            rng = np.random.default_rng(42)
            idx = rng.permutation(14)
            pts = pts[idx]
            inits.append(pts.flatten())
    
    # Random convex configurations
    for seed in range(20):
        rng = np.random.default_rng(seed * 23 + 11)
        # Generate random angles and radii, then sort by angle
        a = np.sort(rng.uniform(0, 2*np.pi, 14))
        r = 0.8 + rng.uniform(0, 0.4, 14)
        pts = np.column_stack([r * np.cos(a), r * np.sin(a)])
        inits.append(pts.flatten())
    
    return inits

# ---------- Optimizers ----------

def _nm_direct(x0, deadline, max_iter=2000, obj=_neg_score_direct):
    remaining = deadline - time.time()
    if remaining < 2.0:
        return None
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        return res
    except Exception:
        return None

def _lbfgs_direct(x0, deadline, max_iter=5000, obj=_neg_score_soft_direct):
    remaining = deadline - time.time()
    if remaining < 5.0:
        return None
    try:
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12})
        return res
    except Exception:
        return None

def _powell_direct(x0, deadline, max_iter=2000, obj=_neg_score_direct):
    remaining = deadline - time.time()
    if remaining < 3.0:
        return None
    try:
        res = minimize(obj, x0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-12, 'ftol': 1e-14})
        return res
    except Exception:
        return None

def _de_direct(bounds, deadline, max_iter=100, popsize=30):
    """Differential evolution on direct 28 parameters."""
    remaining = deadline - time.time()
    if remaining < 20.0:
        return None
    try:
        res = differential_evolution(_neg_score_direct, bounds, maxiter=max_iter,
                                      popsize=popsize, tol=1e-10, atol=1e-12,
                                      seed=42, polish=True, init='latinhypercube')
        return res
    except Exception:
        return None

def _coordinate_descent_polish_direct(points_flat, deadline, max_rounds=50):
    """
    Coordinate descent on the hard objective for final high-precision polish.
    """
    if time.time() > deadline - 1.0:
        return points_flat
    
    best_flat = points_flat.copy()
    best_obj = _neg_score_direct(best_flat)
    
    for _ in range(max_rounds):
        if time.time() > deadline - 1.0:
            break
        improved = False
        for dim in range(28):
            if time.time() > deadline - 1.0:
                break
            step = 1e-5
            for _ in range(15):
                for direction in [-1, 1]:
                    trial = best_flat.copy()
                    trial[dim] += direction * step
                    trial_obj = _neg_score_direct(trial)
                    if trial_obj < best_obj - 1e-15:
                        best_flat = trial
                        best_obj = trial_obj
                        improved = True
                    else:
                        step *= 0.5
                step *= 0.5
                if step < 1e-12:
                    break
        if not improved:
            break
    
    return best_flat

# ---------- Main ----------

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_score = 0.0
    best_pts = None
    best_flat = None

    def _consider_points(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    def _consider_flat(flat):
        nonlocal best_score, best_pts, best_flat
        flat = np.asarray(flat, dtype=float)
        if flat.shape != (28,) or np.any(~np.isfinite(flat)):
            return
        pts = flat.reshape(14, 2)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            best_flat = flat.copy()

    # Phase 1: Global Search with Differential Evolution
    bounds = [(-2.0, 2.0)] * 28
    res = _de_direct(bounds, deadline, max_iter=80, popsize=30)
    if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
        _consider_flat(res.x)

    # Phase 2: Local Refinement from diverse starts
    inits = _generate_direct_inits()
    # Add the best from DE to inits if it exists
    if best_flat is not None:
        inits.insert(0, best_flat)

    for x0 in inits:
        if time.time() > deadline - 45:
            break
        res = _nm_direct(x0, deadline, max_iter=1500, obj=_neg_score_direct)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider_flat(res.x)

    # Phase 3: Gradient-based polish with soft objective
    if best_flat is not None and time.time() < deadline - 25:
        # Try L-BFGS with different betas
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 10:
                break
            obj_soft = lambda p, b=beta: _neg_score_soft_direct(p, beta=b)
            res = _lbfgs_direct(best_flat, deadline, max_iter=2000, obj=obj_soft)
            if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
                _consider_flat(res.x)

    # Phase 4: Final high-precision polish with Powell
    if best_flat is not None and time.time() < deadline - 12:
        res = _powell_direct(best_flat, deadline, max_iter=1500, obj=_neg_score_direct)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider_flat(res.x)

    # Phase 5: Coordinate descent polish for exact local optimum
    if best_flat is not None and time.time() < deadline - 6:
        polished = _coordinate_descent_polish_direct(best_flat, deadline, max_rounds=30)
        _consider_flat(polished)

    # Phase 6: Final NM polish with very tight tolerances
    if best_flat is not None and time.time() < deadline - 3:
        res = _nm_direct(best_flat, deadline, max_iter=500, obj=_neg_score_direct)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider_flat(res.x)

    # Fallback: regular 14-gon
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    final_points = _normalize_unit_area(best_pts)

    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))

    # Final validation
    try:
        final_score = _score(final_points)
        if final_score <= 0:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))

    return final_points
# EVOLVE-BLOCK-END
