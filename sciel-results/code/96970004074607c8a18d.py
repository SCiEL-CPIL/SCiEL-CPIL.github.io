"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute triangle indices
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)  # 286

def _all_tri_areas(pts):
    """Compute all 286 triangle areas for a point set."""
    pts = np.asarray(pts, dtype=np.float64)
    areas = np.empty(NUM_TRIS)
    for idx, (i, j, k) in enumerate(TRI_INDICES):
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        areas[idx] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 0.0
    areas = _all_tri_areas(pts)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area

def _soft_min_ratio(pts, temperature):
    """
    Smooth surrogate: soft-min of triangle areas / hull area.
    Uses log-sum-exp for numerical stability.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return -1e9
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return -1e9
    except Exception:
        return -1e9
    
    areas = _all_tri_areas(pts)
    areas = np.maximum(areas, 1e-15)
    
    # soft_min = -T * log(sum(exp(-a/T)))
    scaled = -areas / temperature
    scaled_max = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(scaled - scaled_max))) + scaled_max
    soft_min_area = -log_sum * temperature
    
    return soft_min_area / hull_area

def _regular_polygon(n, radius=1.0, center=(0,0)):
    """Generate regular polygon vertices."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius*np.cos(angles), 
                            center[1] + radius*np.sin(angles)])

def _radii_to_points(radii, base_radius=1.0):
    """
    Convert radial offsets to Cartesian points.
    radii: array of shape (N,) representing R_i = base_radius + delta_i
    The angles are fixed to those of a regular N-gon.
    """
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    # radii here are the actual radii, not offsets, in this helper
    # But for optimization, we pass offsets. Let's make a specific function for offsets.
    pass

def _offsets_to_points(offsets):
    """
    Convert radial offsets to Cartesian points.
    offsets: array of shape (N,) representing delta_i
    Point i is at radius (1 + delta_i) and angle theta_i of regular N-gon.
    """
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii = 1.0 + offsets
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _neg_smooth_ratio_from_offsets(offsets, temperature):
    """
    Negative of the smooth ratio for minimization, given radial offsets.
    """
    pts = _offsets_to_points(offsets)
    ratio = _soft_min_ratio(pts, temperature)
    if ratio < -1e8:
        return 1e8
    return -ratio

def _neg_exact_ratio_from_offsets(offsets):
    """
    Negative of the exact score for minimization, given radial offsets.
    """
    pts = _offsets_to_points(offsets)
    return -_score(pts)

def _optimize_radial(offsets0, time_budget, rng):
    """
    Optimize the radial offsets using L-BFGS-B with smooth surrogate,
    followed by local search and Nelder-Mead polish.
    """
    start_time = time.time()
    best_offsets = offsets0.copy()
    best_score = _score(_offsets_to_points(best_offsets))
    
    # Stage 1: L-BFGS-B with decreasing temperature
    # We want to maximize min_area / hull_area.
    # The hull area is also a function of offsets.
    # Since angles are fixed, hull area is roughly constant but varies slightly with radii.
    
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002]
    x0 = best_offsets.copy()
    
    # Bounds for offsets to ensure convexity and reasonable size
    # If radii vary too much, points might get close to center or each other.
    # Regular polygon radius is 1. Let's allow offsets in [-0.2, 0.2] initially.
    bounds = [(-0.2, 0.2)] * N
    
    for temp in temperatures:
        if time.time() - start_time > time_budget - 3:
            break
        
        def obj(x):
            return _neg_smooth_ratio_from_offsets(x, temp)
        
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8,
                         'maxls': 40}
            )
            cand = result.x
            cand_score = _score(_offsets_to_points(cand))
            if cand_score > best_score:
                best_score = cand_score
                best_offsets = cand.copy()
            x0 = cand
        except Exception:
            pass
    
    # Stage 2: Local search in offset space
    # Simple coordinate-wise search with adaptive step
    if time.time() - start_time < time_budget - 1:
        step = 0.01
        improved = True
        while improved and (time.time() - start_time < time_budget - 1):
            improved = False
            for i in range(N):
                for delta in [-step, step]:
                    trial = best_offsets.copy()
                    trial[i] += delta
                    if np.all(trial >= -0.2) and np.all(trial <= 0.2):
                        trial_score = _score(_offsets_to_points(trial))
                        if trial_score > best_score:
                            best_score = trial_score
                            best_offsets = trial
                            improved = True
            if not improved:
                step *= 0.5
                if step < 1e-5:
                    break
    
    # Stage 3: Nelder-Mead polish in Cartesian space
    # This allows the angles to shift slightly if beneficial, breaking the strict symmetry
    # but starting from the optimized radial configuration.
    if time.time() - start_time < time_budget - 1:
        best_pts = _offsets_to_points(best_offsets)
        
        def exact_obj(x):
            pts = x.reshape(N, 2)
            return -_score(pts)
        
        try:
            result = minimize(
                exact_obj, best_pts.flatten(), method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11,
                         'initial_simplex': 1e-3}
            )
            cand_pts = result.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                # We return the Cartesian points, so we don't need to convert back to offsets
                # unless we want to continue optimizing in radial space.
                # For now, return the best Cartesian points found.
                return cand_pts, cand_score
        except Exception:
            pass
    
    return _offsets_to_points(best_offsets), best_score

def construct():
    """Main entry point: find optimal 13-point configuration."""
    start_time = time.time()
    total_budget = 145.0  # Safety margin for 150s limit
    
    rng = np.random.RandomState(12345)
    
    best_points = None
    best_ratio = -1e9
    
    # Start with regular polygon (zero offsets)
    # And a few jittered radial offsets
    
    starts = []
    
    # 1. Regular polygon (offsets = 0)
    starts.append(np.zeros(N))
    
    # 2. Jittered offsets
    for seed in [1, 2, 3]:
        rng_local = np.random.RandomState(seed)
        offsets = rng_local.normal(0, 0.05, N)
        starts.append(offsets)
    
    # 3. Alternating large/small offsets (breathing mode)
    offsets_breath = np.array([0.1 if i % 2 == 0 else -0.1 for i in range(N)])
    starts.append(offsets_breath)
    
    # 4. Rotational mode
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    offsets_rot = 0.1 * np.cos(angles)
    starts.append(offsets_rot)
    
    # Evaluate initial candidates
    for offsets in starts:
        if time.time() - start_time > total_budget:
            break
        try:
            pts = _offsets_to_points(offsets)
            s = _score(pts)
            if s > best_ratio:
                best_ratio = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Sort starts by initial score (best first)
    scored_starts = []
    for offsets in starts:
        try:
            pts = _offsets_to_points(offsets)
            s = _score(pts)
            scored_starts.append((s, offsets))
        except Exception:
            scored_starts.append((0.0, offsets))
    scored_starts.sort(key=lambda x: -x[0])
    
    n_starts = len(scored_starts)
    base_time = (total_budget - 5.0) / n_starts
    
    # Optimize from each starting point
    for i, (init_score, offsets) in enumerate(scored_starts):
        if time.time() - start_time > total_budget - 2:
            break
        
        remaining_time = total_budget - (time.time() - start_time)
        # Give more time to better initial points
        quality_factor = 1.0 + 0.5 * (init_score / max(best_ratio, 1e-9))
        alloc_time = min(base_time * quality_factor, remaining_time - 1.0)
        if alloc_time < 2.0:
            continue
        
        try:
            opt_pts, opt_score = _optimize_radial(offsets, alloc_time, rng)
            if opt_score > best_ratio:
                best_ratio = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Final refinement on the best found
    if best_points is not None and time.time() - start_time < total_budget - 3:
        final_time = total_budget - (time.time() - start_time) - 1.0
        if final_time > 8.0:
            # Convert best_points back to offsets for the radial optimizer?
            # Or just run Nelder-Mead directly?
            # Let's run a direct Nelder-Mead polish on the Cartesian points
            def exact_obj(x):
                pts = x.reshape(N, 2)
                return -_score(pts)
            
            try:
                result = minimize(
                    exact_obj, best_points.flatten(), method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12,
                             'initial_simplex': 1e-4}
                )
                cand_pts = result.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                if cand_score > best_ratio:
                    best_ratio = cand_score
                    best_points = cand_pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        # Fallback: regular polygon
        best_points = _regular_polygon(N, radius=1.0)
    
    return best_points
# EVOLVE-BLOCK-END
