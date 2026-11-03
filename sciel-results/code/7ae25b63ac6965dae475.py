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
    f = np.zeros(n)
    for i in range(n):
        f[i] = 1.0
        idx = i + 1
        while idx > 0:
            f[i] /= base
            idx = (idx - 1) // base
        result[i] = f[i]
    return result

def _generate_halton_points(n, num_bases=2):
    """Generate n points in [0,1)^2 using Halton sequences with different bases."""
    # Use first few prime bases
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
    
    # Check for degeneracy
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return -1e9, 0.0, 0.0
    
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area < 1e-12:
            return -1e9, 0.0, 0.0
    except Exception:
        return -1e9, 0.0, 0.0
    
    # Compute all triangle areas efficiently
    # For each pair (i,j), compute cross product with all other points
    min_area = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            # Vector from i to j
            v = pts[j] - pts[i]
            # Cross products with all other points k
            # Area = 0.5 * |cross(v, pts[k] - pts[i])|
            diffs = pts - pts[i]
            cross = v[0] * diffs[:, 1] - v[1] * diffs[:, 0]
            areas = 0.5 * np.abs(cross)
            # Exclude i and j themselves (area would be 0)
            areas[i] = float('inf')
            areas[j] = float('inf')
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    
    if min_area <= 0:
        return -1e9, 0.0, hull_area
    
    ratio = min_area / hull_area
    return ratio, min_area, hull_area

def _soft_min_triangle_ratio(pts, temperature=0.01):
    """
    Smooth surrogate for the min triangle area ratio using log-sum-exp.
    This makes the objective differentiable for optimization.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return -1e9
    
    # Compute hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return -1e9
    except Exception:
        return -1e9
    
    # Compute all triangle areas efficiently
    # There are C(13,3) = 286 triangles
    # For each pair (i,j), compute cross with all k
    all_areas = []
    for i in range(N):
        for j in range(i + 1, N):
            v = pts[j] - pts[i]
            diffs = pts - pts[i]
            cross = v[0] * diffs[:, 1] - v[1] * diffs[:, 0]
            a = 0.5 * np.abs(cross)
            # Exclude i and j
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            mask[j] = False
            valid_a = a[mask]
            all_areas.extend(valid_a)
    
    valid_areas = np.array(all_areas)
    valid_areas = valid_areas[valid_areas > 1e-15]
    
    if len(valid_areas) == 0:
        return -1e9
    
    # Soft min using log-sum-exp
    # soft_min(x) = -log(sum(exp(-x/T))) * T
    scaled = -valid_areas / temperature
    # For numerical stability, subtract max
    scaled_max = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(scaled - scaled_max))) + scaled_max
    soft_min_area = -log_sum * temperature
    
    ratio = soft_min_area / hull_area
    return ratio

def _objective_flat(x, temperature=0.01):
    """Objective function for flat parameter vector (minimize negative ratio)."""
    pts = x.reshape(N, 2)
    # Use soft min for smoothness
    ratio = _soft_min_triangle_ratio(pts, temperature)
    if ratio < -1e8:
        return 1e8
    return -ratio

def _construct_single_start(initial_pts, time_budget, rng):
    """Optimize from a single starting point."""
    start_time = time.time()
    
    # Flatten for optimization
    x0 = initial_pts.flatten()
    
    # Use L-BFGS-B with soft min objective
    best_x = x0.copy()
    best_score = _soft_min_triangle_ratio(initial_pts, temperature=0.01)
    
    # Try optimization with different temperatures (annealing)
    # Use more aggressive schedule with more iterations at lower temps
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002]
    
    for temp in temperatures:
        if time.time() - start_time > time_budget - 3:
            break
        
        def obj(x):
            return _objective_flat(x, temperature=temp)
        
        try:
            # More iterations for lower temperatures
            if temp >= 0.02:
                maxiter = 150
            else:
                maxiter = 300
            
            result = minimize(
                obj, x0, method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < 1e7:  # Valid result
                candidate_pts = result.x.reshape(N, 2)
                candidate_ratio = _soft_min_triangle_ratio(candidate_pts, temperature=temp)
                
                if candidate_ratio > best_score:
                    best_score = candidate_ratio
                    best_x = result.x.copy()
            
            x0 = result.x.copy()
        except Exception:
            pass
    
    # Final polish with very low temperature and tight tolerances
    if time.time() - start_time < time_budget - 2:
        try:
            def obj_polish(x):
                return _objective_flat(x, temperature=0.001)
            
            result = minimize(
                obj_polish, x0, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.fun < 1e7:
                candidate_pts = result.x.reshape(N, 2)
                candidate_ratio = _soft_min_triangle_ratio(candidate_pts, temperature=0.001)
                
                if candidate_ratio > best_score:
                    best_score = candidate_ratio
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Final evaluation with exact min
    best_pts = best_x.reshape(N, 2)
    final_ratio, _, _ = _compute_min_tri_area_norm(best_pts)
    
    return best_pts, final_ratio

def construct():
    """Main entry point: find optimal 13-point configuration."""
    start_time = time.time()
    total_budget = 140.0  # Leave 10s safety margin for 150s limit
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_ratio = -1e9
    
    # Generate multiple diverse starting points
    num_starts = 12
    
    for start_idx in range(num_starts):
        if time.time() - start_time > total_budget - 15:
            break
        
        # Create initial points
        if start_idx == 0:
            # Pure Halton
            init_pts = _generate_halton_points(N, num_bases=2)
        elif start_idx == 1:
            # Halton with slight random perturbation
            init_pts = _generate_halton_points(N, num_bases=2)
            init_pts += rng.normal(0, 0.05, size=(N, 2))
        elif start_idx == 2:
            # Halton with larger perturbation
            init_pts = _generate_halton_points(N, num_bases=2)
            init_pts += rng.normal(0, 0.1, size=(N, 2))
        elif start_idx == 3:
            # Different Halton bases
            init_pts = np.column_stack([
                _halton_sequence(N, 5),
                _halton_sequence(N, 7)
            ])
        elif start_idx == 4:
            # Halton with random rotation
            init_pts = _generate_halton_points(N, num_bases=2)
            theta = rng.uniform(0, 2 * np.pi)
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
            init_pts = init_pts @ rot.T
        elif start_idx == 5:
            # Halton with random scaling
            init_pts = _generate_halton_points(N, num_bases=2)
            init_pts *= rng.uniform(0.8, 1.2, size=(N, 1))
        elif start_idx == 6:
            # Mix of Halton and random
            halton_pts = _generate_halton_points(N, num_bases=2)
            random_pts = rng.uniform(0, 1, size=(N, 2))
            init_pts = 0.5 * halton_pts + 0.5 * random_pts
        elif start_idx == 7:
            # Different Halton bases: 3, 5
            init_pts = np.column_stack([
                _halton_sequence(N, 3),
                _halton_sequence(N, 5)
            ])
        elif start_idx == 8:
            # Halton with random affine transform
            init_pts = _generate_halton_points(N, num_bases=2)
            A = rng.uniform(0.5, 1.5, size=(2, 2))
            init_pts = init_pts @ A.T
            init_pts = np.clip(init_pts, 0.01, 0.99)
        elif start_idx == 9:
            # Sobol-like: use different prime bases with scaling
            init_pts = np.column_stack([
                _halton_sequence(N, 2) * 0.9 + 0.05,
                _halton_sequence(N, 11) * 0.9 + 0.05
            ])
        elif start_idx == 10:
            # Halton with small random jitter
            init_pts = _generate_halton_points(N, num_bases=2)
            init_pts += rng.normal(0, 0.02, size=(N, 2))
        else:
            # Pure random
            init_pts = rng.uniform(0, 1, size=(N, 2))
        
        # Clip to [0, 1]
        init_pts = np.clip(init_pts, 0.01, 0.99)
        
        # Evaluate starting point
        start_ratio, _, _ = _compute_min_tri_area_norm(init_pts)
        if start_ratio > best_ratio:
            best_ratio = start_ratio
            best_points = init_pts.copy()
        
        # Optimize from this start
        time_remaining = total_budget - (time.time() - start_time) - 5
        if time_remaining < 2:
            break
        
        opt_pts, opt_ratio = _construct_single_start(init_pts, time_remaining, rng)
        
        if opt_ratio > best_ratio:
            best_ratio = opt_ratio
            best_points = opt_pts.copy()
    
    # If no valid points found, return Halton as fallback
    if best_points is None:
        best_points = _generate_halton_points(N, num_bases=2)
    
    return best_points

# EVOLVE-BLOCK-END
