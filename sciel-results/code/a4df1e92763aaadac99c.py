"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute triangle indices for speed
TRI_INDICES = np.array(list(combinations(range(N), 3)))
NUM_TRIS = len(TRI_INDICES)

def _all_tri_areas(pts):
    """Vectorized computation of all triangle areas."""
    pts = np.asarray(pts, dtype=np.float64)
    i, j, k = TRI_INDICES.T
    v1 = pts[j] - pts[i]
    v2 = pts[k] - pts[i]
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
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
    scaled = -areas / temperature
    scaled_max = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(scaled - scaled_max))) + scaled_max
    soft_min_area = -log_sum * temperature
    return soft_min_area / hull_area

def _neg_soft_min(pts, temperature):
    """Negative soft-min ratio for minimization."""
    r = _soft_min_ratio(pts, temperature)
    if r < -1e8:
        return 1e8
    return -r

def _neg_exact(pts):
    """Negative exact score for minimization."""
    return -_score(pts)

def _regular_polygon(n, radius=1.0):
    """Generate regular polygon vertices."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _concentric_rings(n, n_rings=3):
    """Generate points on concentric rings."""
    pts = [[0.0, 0.0]]
    remaining = n - 1
    for ring in range(n_rings):
        if remaining <= 0:
            break
        n_on_ring = min(remaining, max(3, (n - 1) // n_rings))
        if ring == n_rings - 1:
            n_on_ring = remaining
        radius = (ring + 1) / n_rings
        angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False) + ring * np.pi / 6
        ring_pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        pts.append(ring_pts)
        remaining -= n_on_ring
    if remaining > 0:
        angles = np.linspace(0, 2 * np.pi, remaining, endpoint=False)
        ring_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        pts.append(ring_pts)
    return np.vstack(pts)[:n]

def _halton_sequence(n, base):
    """Generate n Halton sequence points in [0,1)."""
    result = []
    f = 1.0
    for i in range(1, n + 1):
        x = 0.0
        i_val = i
        while i_val > 0:
            i_val, digit = divmod(i_val, base)
            x += digit * f
            f /= base
        result.append(x)
    return np.array(result)

def _generate_halton_points(n, base1=2, base2=3):
    """Generate n points using 2D Halton sequence mapped to disk."""
    x = _halton_sequence(n, base1)
    y = _halton_sequence(n, base2)
    r = np.sqrt(x)
    theta = 2 * np.pi * y
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

def _local_search_angular(angles, radii, time_budget):
    """
    Coordinate-wise local search in angular parameterization.
    Perturb one angle or radius at a time with adaptive step sizes.
    """
    start_time = time.time()
    best_angles = angles.copy()
    best_radii = radii.copy()
    best_pts = np.column_stack([best_radii * np.cos(best_angles), best_radii * np.sin(best_angles)])
    best_score = _score(best_pts)
    
    angle_step = 0.05
    radius_step = 0.05
    
    while time.time() - start_time < time_budget - 0.5:
        improved = False
        for i in range(N):
            # Try perturbing angle
            for delta in [-angle_step, angle_step]:
                trial_angles = best_angles.copy()
                trial_angles[i] += delta
                trial_pts = np.column_stack([best_radii * np.cos(trial_angles), best_radii * np.sin(trial_angles)])
                trial_score = _score(trial_pts)
                if trial_score > best_score:
                    best_angles = trial_angles
                    best_score = trial_score
                    best_pts = trial_pts
                    improved = True
            
            # Try perturbing radius
            for delta in [-radius_step, radius_step]:
                trial_radii = best_radii.copy()
                trial_radii[i] = np.maximum(0.01, trial_radii[i] + delta)
                trial_pts = np.column_stack([trial_radii * np.cos(best_angles), trial_radii * np.sin(best_angles)])
                trial_score = _score(trial_pts)
                if trial_score > best_score:
                    best_radii = trial_radii
                    best_score = trial_score
                    best_pts = trial_pts
                    improved = True
        
        if not improved:
            angle_step *= 0.5
            radius_step *= 0.5
            if angle_step < 1e-5:
                break
    
    return best_pts, best_score

def _optimize_from_start(initial_pts, time_budget):
    """
    Multi-stage optimization from a starting point:
    1. L-BFGS with smooth soft-min surrogate
    2. Coordinate-wise local search in angular space
    3. Nelder-Mead on exact objective
    """
    start_time = time.time()
    best_pts = initial_pts.copy()
    best_score = _score(initial_pts)
    
    # Convert to angular parameterization
    radii = np.linalg.norm(best_pts, axis=1)
    angles = np.arctan2(best_pts[:, 1], best_pts[:, 0])
    angles = angles % (2 * np.pi)
    
    # Stage 1: L-BFGS with smooth surrogate (decreasing temperature)
    for temp in [0.1, 0.05, 0.02, 0.01]:
        if time.time() - start_time > time_budget - 10:
            break
        try:
            def obj(x):
                pts = x.reshape(-1, 2)
                return _neg_soft_min(pts, temp)
            
            result = minimize(
                obj,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < 1e5:
                pts_opt = result.x.reshape(-1, 2)
                s = _score(pts_opt)
                if s > best_score:
                    best_score = s
                    best_pts = pts_opt.copy()
        except Exception:
            pass
    
    # Stage 2: Coordinate-wise local search in angular space
    if time.time() - start_time < time_budget - 5:
        radii = np.linalg.norm(best_pts, axis=1)
        angles = np.arctan2(best_pts[:, 1], best_pts[:, 0]) % (2 * np.pi)
        remaining_time = time_budget - (time.time() - start_time) - 5
        if remaining_time > 1:
            best_pts, best_score = _local_search_angular(angles, radii, remaining_time)
    
    # Stage 3: Nelder-Mead on exact objective
    if time.time() - start_time < time_budget - 1:
        try:
            result = minimize(
                _neg_exact,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < 1e5:
                pts_opt = result.x.reshape(-1, 2)
                s = _score(pts_opt)
                if s > best_score:
                    best_score = s
                    best_pts = pts_opt.copy()
        except Exception:
            pass
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Regular polygon
    candidates.append(_regular_polygon(N, 1.0))
    
    # 2. Concentric rings
    candidates.append(_concentric_rings(N, 3))
    
    # 3. Halton sequence
    candidates.append(_generate_halton_points(N))
    
    # 4-8. Random points with different seeds
    rng = np.random.default_rng(42)
    for seed in range(5):
        rng_s = np.random.default_rng(seed)
        r = rng_s.uniform(0.1, 1.0, N)
        theta = rng_s.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        candidates.append(pts)
    
    # Evaluate all candidates and pick the best
    best_pts = None
    best_score = -np.inf
    for cand in candidates:
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_pts = cand.copy()
    
    # Run DE to find better starting points
    if time.time() < deadline - 60:
        def _decode_params(params):
            log_r = params[:N]
            theta = params[N:]
            sort_idx = np.argsort(theta)
            theta_sorted = theta[sort_idx]
            log_r_sorted = log_r[sort_idx]
            r = np.exp(log_r_sorted)
            x = r * np.cos(theta_sorted)
            y = r * np.sin(theta_sorted)
            return np.column_stack([x, y])
        
        def _objective(params):
            try:
                pts = _decode_params(params)
                if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
                    return 1e6
                min_dist = np.inf
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.linalg.norm(pts[i] - pts[j])
                        if d < min_dist:
                            min_dist = d
                if min_dist < 1e-6:
                    return 1e6
                hull_area = _convex_hull_area(pts)
                if hull_area < 1e-10:
                    return 1e6
                areas = _all_tri_areas(pts)
                min_area = np.min(areas)
                return -(min_area / hull_area)
            except Exception:
                return 1e6
        
        bounds = [(-2.0, 0.5)] * N + [(0, 2 * np.pi)] * N
        
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=30,
                popsize=10,
                tol=1e-5,
                updating='deferred',
                polish=False
            )
            if de_result.fun < 1e5:
                de_pts = _decode_params(de_result.x)
                de_score = _score(de_pts)
                if de_score > best_score:
                    best_score = de_score
                    best_pts = de_pts.copy()
        except Exception:
            pass
    
    # Multi-stage local optimization from the best starting point
    if best_score > -np.inf and time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 5
        if remaining_time > 10:
            best_pts, best_score = _optimize_from_start(best_pts, remaining_time)
    
    # Final fallback
    if best_score == -np.inf:
        best_pts = _regular_polygon(N, 1.0)
        best_score = _score(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
