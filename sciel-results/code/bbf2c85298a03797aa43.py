"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys

N = 13

# EVOLVE-BLOCK-START

def _halton_sequence(n, base):
    """Generate n points from the Halton sequence in [0,1)."""
    result = np.zeros(n)
    for i in range(n):
        f = 1.0
        idx = i + 1
        while idx > 0:
            f /= base
            idx = (idx - 1) // base
        result[i] = f
    return result

def _generate_halton_points(n, num_bases=2):
    """Generate n points in [0,1)^2 using Halton sequences with different bases."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    points = np.zeros((n, num_bases))
    for d in range(num_bases):
        points[:, d] = _halton_sequence(n, primes[d])
    return points

def _compute_min_tri_area_norm(pts):
    """
    Compute the minimum area of any triangle formed by pts,
    normalized by the area of the convex hull of pts.
    Returns (min_area_ratio, min_area, hull_area) or (-1e9, 0, 0) if invalid.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return -1e9, 0.0, 0.0
    
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return -1e9, 0.0, 0.0
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return -1e9, 0.0, 0.0
    except Exception:
        return -1e9, 0.0, 0.0
    
    min_area = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            v = pts[j] - pts[i]
            diffs = pts - pts[i]
            cross = v[0] * diffs[:, 1] - v[1] * diffs[:, 0]
            areas = 0.5 * np.abs(cross)
            areas[i] = float('inf')
            areas[j] = float('inf')
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    
    if min_area <= 0:
        return -1e9, 0.0, hull_area
    
    ratio = min_area / hull_area
    return ratio, min_area, hull_area

def _soft_min_triangle_ratio(pts, beta=100.0):
    """
    Smooth surrogate for the min triangle area ratio using log-sum-exp.
    Uses beta as inverse temperature (higher = sharper min).
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
    
    all_areas = []
    for i in range(N):
        for j in range(i + 1, N):
            v = pts[j] - pts[i]
            diffs = pts - pts[i]
            cross = v[0] * diffs[:, 1] - v[1] * diffs[:, 0]
            a = 0.5 * np.abs(cross)
            for k in range(N):
                if k != i and k != j:
                    all_areas.append(a[k])
    
    valid_areas = np.array(all_areas)
    valid_areas = valid_areas[valid_areas > 1e-15]
    
    if len(valid_areas) == 0:
        return -1e9
    
    # Soft min: -1/beta * log(sum(exp(-beta * x)))
    scaled = -beta * valid_areas
    scaled_max = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(scaled - scaled_max))) + scaled_max
    soft_min_area = -log_sum / beta
    
    ratio = soft_min_area / hull_area
    return ratio

def _objective_flat(x, beta=100.0):
    """Objective function for flat parameter vector (minimize negative ratio)."""
    pts = x.reshape(N, 2)
    ratio = _soft_min_triangle_ratio(pts, beta=beta)
    if ratio < -1e8:
        return 1e8
    return -ratio

def _construct_single_start(initial_pts, time_budget, rng):
    """Optimize from a single starting point with annealing schedule."""
    start_time = time.time()
    
    x0 = initial_pts.flatten()
    
    best_x = x0.copy()
    best_score = _soft_min_triangle_ratio(initial_pts, beta=100.0)
    
    # Annealing schedule: start with soft min, sharpen gradually
    betas = [10.0, 30.0, 80.0, 200.0, 500.0]
    
    for beta in betas:
        if time.time() - start_time > time_budget - 5:
            break
        
        def obj(x):
            return _objective_flat(x, beta=beta)
        
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < 1e7:
                candidate_pts = result.x.reshape(N, 2)
                candidate_ratio = _soft_min_triangle_ratio(candidate_pts, beta=beta)
                
                if candidate_ratio > best_score:
                    best_score = candidate_ratio
                    best_x = result.x.copy()
            
            x0 = result.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() - start_time < time_budget - 2:
        try:
            def obj_polish(x):
                return _objective_flat(x, beta=1000.0)
            
            result_polish = minimize(
                obj_polish, x0, method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result_polish.fun < 1e7:
                candidate_pts = result_polish.x.reshape(N, 2)
                candidate_ratio = _soft_min_triangle_ratio(candidate_pts, beta=1000.0)
                
                if candidate_ratio > best_score:
                    best_score = candidate_ratio
                    best_x = result_polish.x.copy()
        except Exception:
            pass
    
    best_pts = best_x.reshape(N, 2)
    final_ratio, _, _ = _compute_min_tri_area_norm(best_pts)
    
    return best_pts, final_ratio

def construct():
    """Main entry point: find optimal 13-point configuration."""
    start_time = time.time()
    total_budget = 145.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_ratio = -1e9
    
    # Generate diverse starting points
    starts = []
    
    # 1. Regular 13-gon (centered)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.cos(angles)
    y0 = np.sin(angles)
    starts.append(np.column_stack((x0, y0)))
    
    # 2. Halton sequence points
    halton_pts = _generate_halton_points(N, num_bases=2)
    starts.append(halton_pts)
    
    # 3. Halton centered and normalized
    halton_centered = halton_pts - 0.5
    norms = np.linalg.norm(halton_centered, axis=1)
    if np.max(norms) > 0:
        halton_centered = halton_centered / np.max(norms)
    starts.append(halton_centered)
    
    # 4-13: Perturbed regular polygons with varying noise
    base_poly = np.column_stack((x0, y0))
    for noise_scale in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        noise = rng.normal(0, noise_scale, size=(N, 2))
        p_pert = base_poly + noise
        p_pert = p_pert - np.mean(p_pert, axis=0)
        starts.append(p_pert)
    
    # 14-16: Different Halton bases
    for b1, b2 in [(3, 5), (5, 7), (2, 5)]:
        init_pts = np.column_stack([
            _halton_sequence(N, b1),
            _halton_sequence(N, b2)
        ])
        starts.append(init_pts)
    
    # 17-19: Halton with transformations
    for theta in [0.3, 0.7, 1.1]:
        halton_t = _generate_halton_points(N, num_bases=2)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        starts.append(halton_t @ rot.T)
    
    # 20: Pure random
    starts.append(rng.uniform(0, 1, size=(N, 2)))
    
    num_starts = len(starts)
    
    for start_idx in range(num_starts):
        if time.time() - start_time > total_budget - 10:
            break
        
        init_pts = starts[start_idx].copy()
        
        # Evaluate starting point
        start_ratio, _, _ = _compute_min_tri_area_norm(init_pts)
        if start_ratio > best_ratio:
            best_ratio = start_ratio
            best_points = init_pts.copy()
        
        # Optimize from this start
        time_remaining = total_budget - (time.time() - start_time) - 5
        if time_remaining < 3:
            break
        
        opt_pts, opt_ratio = _construct_single_start(init_pts, time_remaining, rng)
        
        if opt_ratio > best_ratio:
            best_ratio = opt_ratio
            best_points = opt_pts.copy()
    
    # If no valid points found, return regular polygon as fallback
    if best_points is None:
        best_points = np.column_stack((x0, y0))
    
    return best_points

# EVOLVE-BLOCK-END
