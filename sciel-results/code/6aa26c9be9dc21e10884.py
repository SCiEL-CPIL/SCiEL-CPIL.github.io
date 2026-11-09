import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area_fast(points):
    """
    Fast calculation of minimum triangle area using vectorized operations.
    Points: (N, 2) array. Returns minimum area (not doubled).
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            v1 = points[j] - pi
            v2s = points[j + 1:] - pi
            cross_vals = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
            areas = 0.5 * np.abs(cross_vals)
            if len(areas) > 0:
                cur_min = np.min(areas)
                if cur_min < min_area:
                    min_area = cur_min
    return min_area

def _score(points):
    """
    Score = Min Triangle Area / Convex Hull Area.
    """
    try:
        if np.allclose(points, points[0]):
            return 0.0
        
        hull = ConvexHull(points)
        hull_area = hull.volume
        
        if hull_area < 1e-12:
            return 0.0
            
        min_tri_area = _min_triangle_area_fast(points)
        
        if min_tri_area < 1e-12:
            return 0.0
            
        return min_tri_area / hull_area
    except Exception:
        return 0.0

def _negative_score(points):
    """
    Objective function for minimizer.
    """
    return -_score(points)

def _regular_polygon(n, radius=1.0):
    """Generate regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _normalize_points(points):
    """
    Normalize points to have centroid at origin and unit variance (or similar)
    to prevent drift, although score is invariant.
    This helps numerical stability of the optimizer.
    """
    # Center
    points = points - np.mean(points, axis=0)
    # Scale to have mean squared distance 1
    dist_sq = np.sum(points**2, axis=1)
    mean_dist_sq = np.mean(dist_sq)
    if mean_dist_sq > 1e-12:
        points = points / np.sqrt(mean_dist_sq)
    return points

def _jittered_polygon(rng, base_points, sigma=0.1):
    """
    Create a jittered version of the polygon.
    """
    points = base_points.copy()
    # Jitter angles and radii
    angles = np.arctan2(points[:, 1], points[:, 0])
    radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    
    # Add noise
    angles += rng.normal(0, sigma, N)
    radii += rng.normal(0, sigma, N)
    radii = np.abs(radii)
    
    # Sort by angle to maintain convexity order
    sort_idx = np.argsort(angles)
    angles = angles[sort_idx]
    radii = radii[sort_idx]
    
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _random_convex_points(rng):
    """
    Generate random points in convex position.
    """
    angles = np.sort(rng.uniform(0, 2 * np.pi, N))
    # Ensure distinct angles
    angles += rng.uniform(0, 0.1, N)
    angles = np.sort(angles)
    
    # Radii near 1
    radii = 1.0 + rng.normal(0, 0.1, N)
    radii = np.abs(radii)
    
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    def update_best(points, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_points = points.copy()
            best_score = score

    rng = np.random.default_rng(42)
    
    # Initialize a diverse population
    # 1. Regular polygon
    # 2. Jittered regular polygons
    # 3. Random convex points
    
    pop_size = 50
    population = []
    
    # Add regular polygon
    reg_poly = _regular_polygon(N)
    population.append(_normalize_points(reg_poly).flatten())
    
    # Add jittered versions
    for i in range(20):
        jitter = _jittered_polygon(rng, reg_poly, sigma=0.05 + 0.02*i)
        population.append(_normalize_points(jitter).flatten())
        
    # Add random convex points
    for i in range(pop_size - 21):
        if time.time() > deadline - 5:
            break
        rc_points = _random_convex_points(rng)
        population.append(_normalize_points(rc_points).flatten())
        
    # Ensure population size
    while len(population) < pop_size:
        rc_points = _random_convex_points(rng)
        population.append(_normalize_points(rc_points).flatten())
        
    # Evaluate initial population
    scores = []
    for i, p in enumerate(population):
        pts = p.reshape(N, 2)
        s = _score(pts)
        scores.append(s)
        if s > best_score:
            update_best(pts, s)
            
    scores = np.array(scores)
    
    # Use CMA-ES for global optimization
    # CMA-ES is robust for this type of problem
    from cma import CMAESOptimizer
    
    # We will run multiple short CMA-ES runs from the best initial points
    # or one longer run with a large population
    
    # Strategy: Run CMA-ES on the best few initial points
    # Sort population by score
    sorted_indices = np.argsort(scores)[::-1]
    top_k = 5
    
    # We have about 130 seconds left.
    # Let's run CMA-ES on the single best point first, with a large budget.
    # If time permits, run on the second best, etc.
    
    # Actually, a single CMA-ES run with a good initial population (if supported)
    # or just the best point is standard.
    # cma.py supports initial population via 'popsize' but starts from x0.
    # We can use 'bounds' to keep points in a reasonable range? 
    # No, normalization handles scale.
    
    # Let's use a wrapper that enforces normalization during optimization?
    # No, CMA-ES works better on raw coordinates if we fix the scale via normalization
    # in the objective, but that makes the landscape discontinuous?
    # Actually, _score is invariant to scale, so the landscape is flat along the scale direction.
    # This can confuse CMA-ES.
    # Better to fix the scale in the parameterization?
    # The current _score normalizes internally? No, _score does NOT normalize.
    # Wait, _score(points) uses the raw hull area of the input points.
    # If I pass normalized points, the score is consistent.
    # If CMA-ES changes the scale, the score doesn't change (scale invariant),
    # but the convex hull area changes?
    # Score = MinTriArea / HullArea.
    # If we scale points by k, MinTriArea scales by k^2, HullArea scales by k^2.
    # Ratio is invariant.
    # So the objective function is flat in the radial direction.
    # This is bad for CMA-ES.
    
    # Solution: Constrain the optimization to a fixed scale.
    # We can do this by normalizing the points inside the objective function 
    # BEFORE calculating the score, but that creates a discontinuity at 0?
    # No, normalization is smooth except at 0.
    # However, if we normalize inside the objective, the optimizer sees a function
    # that is constant along rays from the origin.
    # This effectively reduces the dimensionality.
    # CMA-ES might struggle with this.
    
    # Alternative: Fix the scale by constraining one point or using a parametrization
    # that doesn't include scale.
    # E.g., fix the first point at (1, 0) and optimize the rest?
    # Or optimize angles and relative radii?
    
    # Let's try a different approach:
    # Use the existing _normalize_points in the objective function passed to CMA-ES.
    # Even if it's flat, CMA-ES might still find the shape optimum.
    # To help, we can add a small penalty for deviation from unit scale?
    # Or just let it run.
    
    def _objective_cma(x):
        pts = x.reshape(N, 2)
        # Normalize to unit scale to remove flat direction
        pts_norm = _normalize_points(pts)
        return -_score(pts_norm)
        
    # Run CMA-ES on the best starting point
    best_start_idx = sorted_indices[0]
    x0 = population[best_start_idx]
    
    # CMA-ES options
    maxiter = 500
    timeout = 120
    
    try:
        opts = {
            'maxiter': maxiter,
            'bounds': [-5, 5],  # Loose bounds, normalization handles rest
            'verbose': -9,
            'seed': 42
        }
        
        # We need to monitor time
        # cma.py does not have a built-in timeout callback in older versions?
        # We can use a custom callback or just run it and check time?
        # If it runs too long, we kill it?
        # Better to use a smaller maxiter and check time after each run?
        # Or use a thread?
        
        # Let's just run it with a reasonable maxiter and hope it finishes.
        # 28 dimensions, 500 iterations might take a while.
        # Let's estimate: 500 * 50 (popsize) * cost_of_score.
        # Cost of score is ~ N^3 = 14^3 = 2744 operations. Very fast.
        # 500 * 50 = 25000 evaluations.
        # 25000 * 0.001s = 25s. Should be fine.
        
        es = CMAESOptimizer(x0, 0.5, _objective_cma, opts)
        
        while es.active() and time.time() < deadline - 10:
            es.step()
            
        # Get best point
        if es.result is not None:
            x_best = es.result.xbest
            pts_best = x_best.reshape(N, 2)
            s_best = _score(_normalize_points(pts_best))
            if s_best > best_score:
                update_best(_normalize_points(pts_best), s_best)
                
    except Exception as e:
        pass
        
    # If we have time, run CMA-ES on the second best start
    if time.time() < deadline - 20:
        best_start_idx = sorted_indices[1]
        x0 = population[best_start_idx]
        try:
            opts = {
                'maxiter': 200,
                'bounds': [-5, 5],
                'verbose': -9,
                'seed': 43
            }
            es = CMAESOptimizer(x0, 0.5, _objective_cma, opts)
            while es.active() and time.time() < deadline - 5:
                es.step()
            if es.result is not None:
                x_best = es.result.xbest
                pts_best = x_best.reshape(N, 2)
                s_best = _score(_normalize_points(pts_best))
                if s_best > best_score:
                    update_best(_normalize_points(pts_best), s_best)
        except Exception:
            pass

    # Final local polish with Nelder-Mead on the best CMA-ES result
    if best_points is not None and time.time() < deadline - 5:
        from scipy.optimize import minimize
        
        def _neg_score_general(params):
            pts = params.reshape(N, 2)
            return -_score(_normalize_points(pts))
            
        x0 = _normalize_points(best_points).flatten()
        try:
            res = minimize(_neg_score_general, x0, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
            pts_refined = _normalize_points(res.x.reshape(N, 2))
            s_refined = _score(pts_refined)
            if s_refined > best_score:
                update_best(pts_refined, s_refined)
        except Exception:
            pass

    if best_points is None:
        best_points = reg_poly
        
    return best_points

# EVOLVE-BLOCK-END
