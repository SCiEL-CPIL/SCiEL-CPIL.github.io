import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time

N = 14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    # Check for degenerate cases
    if points.shape != (N, 2):
        return 0.0
    
    # Center the points to avoid numerical issues with large coordinates
    # and to enforce translation invariance implicitly in the search space
    # if we were to constrain it, but here we just center for stability.
    center = np.mean(points, axis=0)
    points = points - center
    
    # Check if points are too close to each other (degenerate)
    # If any pair is very close, triangle areas will be small.
    # A simple check: if hull area is 0, return 0.
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    # Extract points for all triples
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    # Compute cross product
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _negative_score(points_flat):
    """
    Objective function for optimization.
    Returns negative score to be minimized.
    """
    points = points_flat.reshape(N, 2)
    score = _calculate_score(points)
    if score == 0.0:
        # Penalize degenerate configurations heavily
        return 1e6
    return -score

def _generate_regular_polygon(r=1.0):
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return np.column_stack((x, y))

def _polish(points, time_limit):
    """
    Perform local optimization (Nelder-Mead) on the points.
    """
    start_time = time.time()
    best_score = _calculate_score(points)
    best_points = points.copy()
    
    x0 = points.flatten()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return 1e9
        return _negative_score(x)
        
    try:
        res = minimize(
            _negative_score, 
            x0, 
            method='Nelder-Mead', 
            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000, 'disp': False}
        )
        
        if time.time() - start_time < time_limit:
             new_points = res.x.reshape(N, 2)
             new_score = _calculate_score(new_points)
             if new_score > best_score:
                 best_score = new_score
                 best_points = new_points
                 
    except Exception:
        pass
        
    return best_score, best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = 0.0
    best_points = _generate_regular_polygon(1.0)
    
    # 1. Try Differential Evolution
    # Bounds for x and y
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # We want to run DE for a significant amount of time.
    # differential_evolution doesn't have a direct time limit, but we can
    # use maxiter and popsize to control it roughly, or just run it and hope
    # it finishes. A better way is to use a callback or just set a reasonable
    # maxiter.
    
    # Let's estimate time per iteration.
    # One evaluation of _negative_score takes ~1-2ms.
    # DE with popsize=15 does 15 evaluations per generation.
    # If we allow 100 generations, that's 1500 evals ~ 3s.
    # We have ~140s. So we can do many more.
    # Let's try maxiter=500, popsize=20.
    # 500 * 20 * 15 (evals per gen approx) = 150,000 evals.
    # 150,000 * 1ms = 150s. This is risky.
    
    # Let's be conservative. maxiter=200, popsize=15.
    # 200 * 15 * 15 = 45,000 evals ~ 45s.
    
    # We can run multiple DE runs if time permits.
    
    try:
        # Set seed for reproducibility if needed, but DE is stochastic.
        res_de = differential_evolution(
            _negative_score,
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False # We will polish manually later
        )
        
        if res_de.success or res_de.fun < 0:
            de_points = res_de.x.reshape(N, 2)
            de_score = _calculate_score(de_points)
            if de_score > best_score:
                best_score = de_score
                best_points = de_points
    except Exception as e:
        pass

    if time.time() > deadline - 20:
        # Not enough time for polishing, return best found
        return best_points

    # 2. Polish the best candidate from DE
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        score, pts = _polish(best_points, remaining_time - 5)
        if score > best_score:
            best_score = score
            best_points = pts

    # 3. If time remains, try another DE run with different seed or params
    if time.time() < deadline - 30:
        try:
            res_de2 = differential_evolution(
                _negative_score,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=123,
                polish=False
            )
            
            if res_de2.success or res_de2.fun < 0:
                de_points = res_de2.x.reshape(N, 2)
                de_score = _calculate_score(de_points)
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_points
        except Exception:
            pass
        
        # Polish again if time permits
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            score, pts = _polish(best_points, remaining_time - 5)
            if score > best_score:
                best_score = score
                best_points = pts

    # 4. Final polish with whatever time is left
    remaining_time = deadline - time.time()
    if remaining_time > 1:
        score, pts = _polish(best_points, remaining_time)
        if score > best_score:
            best_score = score
            best_points = pts

    return best_points
