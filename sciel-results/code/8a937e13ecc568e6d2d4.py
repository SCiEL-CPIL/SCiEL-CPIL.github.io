"""Packing 50 equal circles in a unit circle.
Strategy:
1. Parametric Global Search: Model packing as concentric rings with multiple structures.
   Optimize ring radii and rotations using Differential Evolution.
2. Local Polish: Refine all 100 coordinates using L-BFGS-B with a smooth soft-min objective.
3. Exact Polish: Directly optimize the min-distance using active-set refinement.
"""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist

N = 50
TIME_LIMIT = 148.0  # Safety margin before 150s kill

# Multiple ring structures to try
RING_STRUCTURES = [
    [1, 6, 12, 18, 13],   # Original
    [1, 5, 10, 15, 19],   # Alternative
    [1, 6, 11, 17, 15],   # Another alternative
]

def _compute_score(points):
    """Compute the maximum radius for given points."""
    if not np.all(np.isfinite(points)):
        return 0.0
    n = len(points)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    min_pair_r = min_pair / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(norms)
    
    return min(min_pair_r, min_boundary)

def _build_points_from_params(params, ring_sizes=None):
    """
    Construct 50 points from parametric representation.
    params: array of shape (2*(n_rings-1),).
    First (n_rings-1) values: radii for rings 1..n_rings-1 (center is always 0).
    Next (n_rings-1) values: rotation angles for rings 1..n_rings-1.
    """
    if ring_sizes is None:
        ring_sizes = RING_STRUCTURES[0]
    n_rings = len(ring_sizes)
    points = np.zeros((N, 2))
    idx = 0
    
    # Ring 0: Center
    points[idx] = [0.0, 0.0]
    idx += 1
    
    for i in range(1, n_rings):
        count = ring_sizes[i]
        radius = params[i-1]
        angle_offset = params[n_rings - 1 + (i-1)]
        
        for j in range(count):
            angle = 2 * np.pi * j / count + angle_offset
            points[idx, 0] = radius * np.cos(angle)
            points[idx, 1] = radius * np.sin(angle)
            idx += 1
            
    return points

def _parametric_objective(params, ring_sizes=None):
    """
    Objective for differential evolution: minimize negative score.
    """
    points = _build_points_from_params(params, ring_sizes)
    score = _compute_score(points)
    return -score

def _soft_min_surrogate(points, beta=100.0):
    """
    Compute a smooth surrogate for the min of:
    - pairwise distances / 2
    - boundary distances (1 - norm)
    Uses only the K closest pairs to save computation.
    """
    n = len(points)
    if n < 2:
        return 0.0
    
    # Pairwise distances - use cdist for efficiency
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    
    # Get the K smallest pairwise distances to keep computation manageable
    K = min(200, n * (n - 1) // 2)
    flat_dists = dists.flatten()
    # Get indices of K smallest
    part_idx = np.argpartition(flat_dists, K)[:K]
    pair_vals = flat_dists[part_idx] / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    boundary_vals = 1.0 - norms
    
    # Combine all values
    all_vals = np.concatenate([pair_vals, boundary_vals])
    
    # Soft-min via log-sum-exp
    m = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - m))
    soft_min = m - np.log(np.sum(exp_vals)) / beta
    
    return soft_min

def _smooth_objective(x, beta=100.0):
    """
    Objective for L-BFGS-B: minimize negative soft-min radius + penalty.
    """
    pts = x.reshape(N, 2)
    
    # Hard penalty for points outside unit circle
    norms = np.linalg.norm(pts, axis=1)
    boundary_violation = np.maximum(0, norms - 1.0)
    penalty = 100.0 * np.sum(boundary_violation)
    
    # Negative soft-min radius
    radius = _soft_min_surrogate(pts, beta)
    
    return -radius + penalty

def _exact_objective(x):
    """
    Exact objective: minimize negative of the minimum radius.
    This is nonsmooth but we can use it with Nelder-Mead or SLSQP.
    """
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    norms = np.linalg.norm(pts, axis=1)
    if np.max(norms) > 1.0:
        return 1e10
    score = _compute_score(pts)
    return -score

def _polish_with_smooth(points, deadline, betas=None, maxiter=200):
    """
    Polish points using L-BFGS-B with increasing beta.
    """
    if betas is None:
        betas = [30, 60, 120, 250, 500, 1000, 2000]
    
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    for beta in betas:
        if time.time() > deadline - 8:
            break
            
        def obj(x):
            return _smooth_objective(x, beta)
        
        try:
            result = minimize(
                obj,
                current_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
            )
            
            new_points = result.x.reshape(N, 2)
            
            # Check validity
            if np.all(np.isfinite(new_points)):
                norms = np.linalg.norm(new_points, axis=1)
                # Clip points that went slightly outside due to numerical error
                if np.max(norms) > 1.0:
                    mask = norms > 1.0
                    new_points[mask] *= (1.0 - 1e-12) / norms[mask]
                    
                new_score = _compute_score(new_points)
                if new_score > current_score:
                    current_points = new_points
                    current_score = new_score
        except Exception:
            pass
    
    return current_points

def _polish_exact(points, deadline, maxiter=500):
    """
    Exact local polish using Nelder-Mead on the true nonsmooth objective.
    This can find improvements that the smooth surrogate misses.
    """
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    if time.time() > deadline - 5:
        return current_points
    
    try:
        result = minimize(
            _exact_objective,
            current_points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        
        new_points = result.x.reshape(N, 2)
        
        if np.all(np.isfinite(new_points)):
            norms = np.linalg.norm(new_points, axis=1)
            if np.max(norms) > 1.0:
                mask = norms > 1.0
                new_points[mask] *= (1.0 - 1e-12) / norms[mask]
            
            new_score = _compute_score(new_points)
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
    except Exception:
        pass
    
    return current_points

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = None
    best_score = -np.inf
    
    # --- Phase 1: Parametric Global Search over multiple ring structures ---
    
    # Time budget for DE: use about 40% of total time
    de_time_budget = (deadline - start_time) * 0.35
    de_deadline = start_time + de_time_budget
    
    for ring_sizes in RING_STRUCTURES:
        if time.time() > de_deadline:
            break
        
        n_rings = len(ring_sizes)
        n_params = 2 * (n_rings - 1)
        
        # Bounds for parameters:
        # Radii: 0.1 to 0.95
        # Angles: 0 to 2*pi
        bounds = [(0.1, 0.95)] * (n_rings - 1) + [(0.0, 2 * np.pi)] * (n_rings - 1)
        
        def wrapper(params, _ring_sizes=ring_sizes):
            if time.time() > de_deadline:
                raise TimeoutError
            return _parametric_objective(params, _ring_sizes)
        
        try:
            result = differential_evolution(
                wrapper,
                bounds,
                seed=42,
                maxiter=80,
                popsize=12,
                tol=1e-8,
                mutation=(0.4, 1.2),
                recombination=0.8,
                updating='immediate'
            )
            
            if result.x is not None:
                pts = _build_points_from_params(result.x, ring_sizes)
                s = _compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
                    
        except TimeoutError:
            pass
        except Exception:
            pass
    
    # Fallback if DE failed or timed out: use static initializations
    if best_score < 0.1:
        rng = np.random.default_rng(42)
        for ring_sizes in RING_STRUCTURES:
            n_rings = len(ring_sizes)
            # Static heuristic: evenly spaced radii
            radii = np.linspace(0.2, 0.9, n_rings - 1)
            angles = rng.uniform(0, 2*np.pi, n_rings - 1)
            params = np.concatenate([radii, angles])
            pts = _build_points_from_params(params, ring_sizes)
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
    
    if best_points is None:
        # Ultimate fallback
        best_points = np.zeros((N, 2))
        best_score = 0.0
    
    current_points = best_points.copy()
    current_score = best_score
    
    # --- Phase 2: Local Polish with Smooth Surrogate ---
    
    if time.time() < deadline - 15:
        polished = _polish_with_smooth(current_points, deadline, 
                                        betas=[30, 60, 120, 250, 500, 1000], 
                                        maxiter=300)
        polished_score = _compute_score(polished)
        if polished_score > current_score:
            current_points = polished
            current_score = polished_score
    
    # --- Phase 3: Exact Polish ---
    
    if time.time() < deadline - 5:
        polished = _polish_exact(current_points, deadline, maxiter=500)
        polished_score = _compute_score(polished)
        if polished_score > current_score:
            current_points = polished
            current_score = polished_score
    
    # --- Phase 4: Final smooth polish with very high beta ---
    
    if time.time() < deadline - 2:
        polished = _polish_with_smooth(current_points, deadline, 
                                        betas=[2000, 5000], 
                                        maxiter=200)
        polished_score = _compute_score(polished)
        if polished_score > current_score:
            current_points = polished
            current_score = polished_score
    
    # Ensure valid output
    current_points = np.array(current_points, dtype=np.float64)
    norms = np.linalg.norm(current_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        current_points[mask] *= (1.0 - 1e-12) / norms[mask]
    
    return current_points
