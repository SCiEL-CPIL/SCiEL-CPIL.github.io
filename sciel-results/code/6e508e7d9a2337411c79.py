"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

# EVOLVE-BLOCK-START
import random
import sys

def _compute_min_tri_area_norm(pts):
    """
    Compute the minimum area of any triangle formed by pts,
    normalized by the area of the convex hull of pts.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return -1e9
        
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area < 1e-9:
            return -1e9
    except Exception:
        return -1e9

    # Compute all triangle areas
    min_area = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + 
                                  p2[0] * (p3[1] - p1[1]) + 
                                  p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area

    return min_area / hull_area

def _soft_min_triangle_ratio(pts, temperature=0.01):
    """
    Compute a smooth surrogate for the minimum triangle area ratio.
    Uses the log-sum-exp trick to approximate the minimum.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return -1e9
        
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return -1e9
    except Exception:
        return -1e9

    # Compute all triangle areas
    areas = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + 
                                  p2[0] * (p3[1] - p1[1]) + 
                                  p3[0] * (p1[1] - p2[1]))
                areas.append(area)
    
    areas = np.array(areas)
    
    # Log-sum-exp trick for minimum
    # min(x) = -max(-x)
    # log(sum(exp(x/T))) approximates max(x) as T->0
    # So min(x) approx -T * log(sum(exp(-x/T)))
    
    neg_areas = -areas
    max_neg = np.max(neg_areas)
    # Handle numerical stability
    if max_neg > 700:
        max_neg = 700
    sum_exp = np.sum(np.exp(neg_areas - max_neg))
    log_sum_exp = max_neg + np.log(sum_exp)
    
    soft_min = -temperature * log_sum_exp
    
    return soft_min / hull_area

def _objective_flat(params, temperature=0.01):
    pts = params.reshape(N, 2)
    val = _soft_min_triangle_ratio(pts, temperature)
    return -val

def _halton_sequence(index, base):
    f = 1.0
    r = 0.0
    while index > 0:
        f /= base
        r += f * (index % base)
        index //= base
    return r

def _generate_halton_points(n):
    pts = []
    for i in range(1, n + 1):
        x = _halton_sequence(i, 2)
        y = _halton_sequence(i, 3)
        pts.append([x, y])
    pts = np.array(pts)
    # Center
    pts -= np.mean(pts, axis=0)
    return pts

def _generate_diverse_starts(rng, n_starts=10):
    """Generate diverse starting points for optimization."""
    starts = []
    
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg = np.column_stack((np.cos(angles), np.sin(angles)))
    starts.append(reg.flatten())
    
    # Halton sequence
    halton = _generate_halton_points(N)
    starts.append(halton.flatten())
    
    # Random starts
    for _ in range(n_starts - 2):
        rand_pts = rng.random((N, 2)) * 2 - 1
        rand_pts -= np.mean(rand_pts, axis=0)
        starts.append(rand_pts.flatten())
    
    # Perturbed regular polygon
    for _ in range(3):
        noise = rng.normal(0, 0.3, size=N * 2)
        perturbed = reg.flatten() + noise
        perturbed = perturbed.reshape(N, 2)
        perturbed -= np.mean(perturbed, axis=0)
        starts.append(perturbed.flatten())
    
    return starts

def _optimize_from_start(start, temperature=0.01, maxiter=3000):
    """Optimize from a single starting point."""
    try:
        res = minimize(
            _objective_flat, 
            start, 
            args=(temperature,),
            method='L-BFGS-B', 
            options={'maxiter': maxiter, 'maxfun': maxiter * 10, 'disp': False}
        )
        return res.x
    except Exception:
        return start

def _local_search_around(pts, rng, n_tries=5, temperature=0.005):
    """Perform local search around a given point set."""
    best_pts = pts.copy()
    best_score = _compute_min_tri_area_norm(pts)
    
    for _ in range(n_tries):
        # Random perturbation
        noise = rng.normal(0, 0.05, size=N * 2)
        perturbed = pts.flatten() + noise
        perturbed = perturbed.reshape(N, 2)
        perturbed -= np.mean(perturbed, axis=0)
        
        # Optimize from perturbed point
        opt_result = _optimize_from_start(perturbed.flatten(), temperature, maxiter=1000)
        score = _compute_min_tri_area_norm(opt_result.reshape(N, 2))
        
        if score > best_score:
            best_score = score
            best_pts = opt_result.copy()
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin from 150s kill limit
    
    rng = np.random.default_rng(42)
    
    best_score = -1e9
    best_pts = None
    
    # Phase 1: Coarse optimization with multiple starts
    starts = _generate_diverse_starts(rng, n_starts=12)
    
    for i, start in enumerate(starts):
        if time.time() - start_time > time_limit - 20.0:
            break
        
        # Use higher temperature for coarse search
        temp = 0.05 if i < 4 else 0.02
        result = _optimize_from_start(start, temp, maxiter=2000)
        score = _compute_min_tri_area_norm(result.reshape(N, 2))
        
        if score > best_score:
            best_score = score
            best_pts = result.copy()
    
    # Phase 2: Fine optimization with lower temperature
    if best_pts is not None:
        # Continue with the best found so far
        result = _optimize_from_start(best_pts, temperature=0.01, maxiter=3000)
        score = _compute_min_tri_area_norm(result.reshape(N, 2))
        if score > best_score:
            best_score = score
            best_pts = result.copy()
        
        # Try a few more refined starts
        for _ in range(3):
            if time.time() - start_time > time_limit - 10.0:
                break
            noise = rng.normal(0, 0.1, size=N * 2)
            perturbed = best_pts + noise
            perturbed = perturbed.reshape(N, 2)
            perturbed -= np.mean(perturbed, axis=0)
            
            result = _optimize_from_start(perturbed.flatten(), temperature=0.005, maxiter=2000)
            score = _compute_min_tri_area_norm(result.reshape(N, 2))
            if score > best_score:
                best_score = score
                best_pts = result.copy()
    
    # Phase 3: Local search refinement
    if best_pts is not None and time.time() - start_time < time_limit - 5.0:
        best_pts, best_score = _local_search_around(
            best_pts.reshape(N, 2), rng, n_tries=10, temperature=0.003
        )
        best_pts = best_pts.flatten()
    
    if best_pts is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_pts.reshape(N, 2)

# EVOLVE-BLOCK-END
