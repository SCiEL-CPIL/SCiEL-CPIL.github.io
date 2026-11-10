import numpy as np
import scipy.optimize
import time
from scipy.optimize import differential_evolution

def _vertex_score(v, pts):
    """Compute the vertex score for vertex v given all points pts."""
    diffs = pts - v
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    mask = np.ones(len(pts), dtype=bool)
    mask[np.argmin(dists)] = False
    d = np.sort(dists[mask])
    if np.any(d < 0.05):
        return 0.0
    gaps = np.diff(d)
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k - 1:k + 3]
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _min_vertex_score(pts):
    scores = [_vertex_score(pts[i], pts) for i in range(len(pts))]
    return min(scores)

def _is_convex(pts):
    """Check if pts form a strictly convex polygon."""
    n = len(pts)
    crosses = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        crosses.append(v1[0] * v2[1] - v1[1] * v2[0])
    crosses = np.array(crosses)
    if np.any(crosses == 0):
        return False
    if not np.all(crosses > 0) and not np.all(crosses < 0):
        return False
    return True

def _normalize(pts):
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    dists = np.linalg.norm(pts_norm, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return pts_norm, False
    pts_norm = pts_norm / mean_dist
    return pts_norm, True

def _soft_min(scores, temperature=0.01):
    """Smoothed minimum using log-sum-exp."""
    min_val = np.min(scores)
    shifted = scores - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum

def _smooth_objective(coords_flat, temperature=0.01):
    """Smoothed objective: negative soft-min of vertex scores."""
    n = 20
    pts = coords_flat.reshape(n, 2)
    
    # Normalize
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    dists = np.linalg.norm(pts_norm, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return 1000.0
    pts_norm = pts_norm / mean_dist
    
    # Check convexity
    if not _is_convex(pts_norm):
        return 1000.0
    
    # Compute vertex scores
    scores = np.array([_vertex_score(pts_norm[i], pts_norm) for i in range(n)])
    
    # Use soft-min to make objective smooth
    soft_min_val = _soft_min(scores, temperature)
    
    return -soft_min_val

def _polygon_from_radii(radii):
    """Convert a vector of 20 radii to 2D points using fixed angular positions."""
    angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _evaluate_radii(radii):
    """Evaluate the objective for a given set of radii. Returns negative score (to minimize)."""
    pts = _polygon_from_radii(radii)
    pts, valid = _normalize(pts)
    if not valid:
        return 1000.0
    if not _is_convex(pts):
        return 1000.0
    score = _min_vertex_score(pts)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # Initial guess: regular 20-gon (all radii = 1)
    init_radii = np.ones(20)
    best_pts = _polygon_from_radii(init_radii)
    best_pts, _ = _normalize(best_pts)
    best_score = _min_vertex_score(best_pts)
    
    # Phase 1: Differential Evolution in radius space
    try:
        # Bounds: radii must be positive to maintain convexity (roughly)
        # We allow a wide range but DE will constrain
        bounds = [(0.5, 1.5) for _ in range(20)]
        
        # Use a population size that is manageable within time
        # maxiter is the number of generations
        # tol is the convergence tolerance
        
        def de_objective(radii):
            if time.time() > deadline - 180.0:
                return 1000.0
            return _evaluate_radii(radii)
        
        # DE is stochastic, so we run it with a moderate population
        # 'seed' for reproducibility if needed, but let's let it explore
        result = differential_evolution(de_objective, bounds, 
                                         maxiter=100, 
                                         popsize=15,
                                         tol=1e-5,
                                         polish=True,
                                         maxfev=2000,
                                         init='latinhypercube')
        
        if result.fun < -best_score:
            candidate_radii = result.x
            candidate_pts = _polygon_from_radii(candidate_radii)
            candidate_pts, valid = _normalize(candidate_pts)
            if valid and _is_convex(candidate_pts):
                score = _min_vertex_score(candidate_pts)
                if score > best_score:
                    best_score = score
                    best_pts = candidate_pts
                    
    except Exception:
        pass
    
    # Phase 2: Local refinement in radius space using Nelder-Mead
    try:
        if time.time() < deadline - 60.0:
            # Extract current best radii
            # We need to recover radii from best_pts or keep track of them
            # For simplicity, let's re-optimize from the regular polygon if we didn't update,
            # or use the result from DE if available.
            
            # Let's assume we want to refine the best found so far.
            # If DE succeeded, we have candidate_radii. If not, we start from ones.
            # Better: keep track of best_radii.
            
            # Let's refactor slightly to keep best_radii
            # But since we already have best_pts, let's just run a local search on radii
            # starting from the regular polygon or the DE result.
            
            # To be safe, let's run NM starting from the regular polygon and the DE result (if improved)
            
            # For now, let's just run NM from the current best approximation.
            # Since we don't have best_radii explicitly stored if DE failed, let's just use ones.
            # Actually, let's just run a quick NM from ones.
            
            def nm_objective(radii):
                return _evaluate_radii(radii)
            
            # Start from regular
            x0 = np.ones(20)
            result_nm = scipy.optimize.minimize(nm_objective, x0, method='Nelder-Mead',
                                                 options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result_nm.fun < -best_score:
                candidate_radii = result_nm.x
                candidate_pts = _polygon_from_radii(candidate_radii)
                candidate_pts, valid = _normalize(candidate_pts)
                if valid and _is_convex(candidate_pts):
                    score = _min_vertex_score(candidate_pts)
                    if score > best_score:
                        best_score = score
                        best_pts = candidate_pts
                        
    except Exception:
        pass
        
    return best_pts
