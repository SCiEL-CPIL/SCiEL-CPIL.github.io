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
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _soft_min_score(points, beta):
    """
    Smooth surrogate: -soft_min of triangle areas / hull area.
    Returns a smooth value that approximates -min_area/hull_area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    areas = 0.5 * np.abs(cross)
    
    # Soft-min via log-sum-exp
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    soft_min_val = (1.0 / beta) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    
    return soft_min_val / hull_area

def _get_ring_points(n_out, n_in, r, theta):
    """
    Generate points for the 2-parameter ring configuration.
    """
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def _optimize_ring_quick(n_out, n_in, time_budget):
    """
    Quick optimization of ring parameters to get a good starting point.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.0, 0.999), (0.0, theta_max)]
    
    best_score = 0.0
    best_params = None
    
    # Use Differential Evolution with limited iterations
    maxiter = 20
    popsize = 10
    
    try:
        res = differential_evolution(
            lambda p: -_calculate_score(_get_ring_points(n_out, n_in, p[0], p[1])),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=False # Skip polish to save time, we will do coordinate polish later
        )
        
        if res.success:
            best_params = res.x
            best_score = -res.fun
    except Exception:
        pass
    
    if best_params is None:
        # Fallback to heuristic
        best_params = [0.7, theta_max/2]
        best_score = _calculate_score(_get_ring_points(n_out, n_in, best_params[0], best_params[1]))
        
    points = _get_ring_points(n_out, n_in, best_params[0], best_params[1])
    return best_score, points

def _coordinate_anneal(points, deadline):
    """
    Optimize points in 28D coordinate space using L-BFGS-B with beta annealing.
    """
    current_points = points.copy()
    current_score = _calculate_score(current_points)
    
    # Define stages of annealing
    # (beta, maxiter, ftol, gtol)
    stages = [
        (10.0, 100, 1e-12, 1e-8),
        (50.0, 100, 1e-12, 1e-8),
        (200.0, 200, 1e-12, 1e-9),
        (1000.0, 100, 1e-12, 1e-10)
    ]
    
    for beta, maxiter, ftol, gtol in stages:
        if time.time() > deadline - 2.0:
            break
            
        def neg_soft_obj(x):
            pts = x.reshape(-1, 2)
            return _soft_min_score(pts, beta)
        
        try:
            res = minimize(
                neg_soft_obj,
                current_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
            )
            
            new_points = res.x.reshape(-1, 2)
            new_score = _calculate_score(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                
        except Exception:
            pass
            
    return current_points, current_score

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s, we aim for 140s
    deadline = start_time + 140.0
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # 1. Generate Initial Candidates
    
    # Candidate 1: Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    regular_points = np.column_stack((np.cos(angles), np.sin(angles)))
    regular_score = _calculate_score(regular_points)
    if regular_score > best_overall_score:
        best_overall_score = regular_score
        best_overall_points = regular_points
    
    # Candidate 2-4: Ring configurations
    # We test a few promising ring splits
    ring_candidates = [
        (7, 7),
        (8, 6),
        (6, 8),
        (9, 5)
    ]
    
    ring_points_list = []
    
    for n_out, n_in in ring_candidates:
        if time.time() > deadline - 60.0: # Reserve 60s for coordinate annealing
            break
        # Allocate 10s per ring optimization
        score, points = _optimize_ring_quick(n_out, n_in, 10.0)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = points
        ring_points_list.append((score, points))
        
    # 2. Coordinate Annealing on Best Candidates
    
    # Sort ring candidates by score
    ring_points_list.sort(key=lambda x: x[0], reverse=True)
    
    # We will anneal the best ring candidate and the regular polygon
    candidates_to_anneal = []
    
    # Add regular polygon
    candidates_to_anneal.append(regular_points)
    
    # Add top ring candidates
    for score, points in ring_points_list[:3]:
        candidates_to_anneal.append(points)
        
    # Add the current best overall if not already in list (should be one of above)
    if best_overall_points is not None:
        # Check if best_overall_points is already in candidates_to_anneal
        # Simple check: if it's the regular polygon or one of the rings
        # We'll just add it if it's not the first one (regular) to be safe, 
        # but since we added regular and top rings, it should be covered.
        pass
    
    # Perform annealing on each candidate, respecting time budget
    # We have ~140s total. 
    # Ring opt took ~40s (4 * 10s).
    # Remaining: ~100s.
    # We have 4 candidates. ~25s each.
    
    time_per_anneal = 25.0
    
    for i, pts in enumerate(candidates_to_anneal):
        if time.time() > deadline - 5.0:
            break
            
        cand_deadline = min(deadline - 5.0, time.time() + time_per_anneal)
        
        try:
            annealed_points, annealed_score = _coordinate_anneal(pts, cand_deadline)
            
            if annealed_score > best_overall_score:
                best_overall_score = annealed_score
                best_overall_points = annealed_points
        except Exception:
            pass
            
    # 3. Final Polish
    # If we have time, do a final high-precision polish on the best point
    if best_overall_points is not None and time.time() < deadline - 2.0:
        final_deadline = deadline - 1.0
        try:
            final_points, final_score = _coordinate_anneal(best_overall_points, final_deadline)
            if final_score > best_overall_score:
                best_overall_score = final_score
                best_overall_points = final_points
        except Exception:
            pass
            
    # Fallback
    if best_overall_points is None:
        best_overall_points = regular_points
        
    return best_overall_points
