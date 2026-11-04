import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 14

# Precompute indices for all triples
def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

_TRIPLE_INDICES = _get_triple_indices(N)
_N_TRIPLES = len(_TRIPLE_INDICES)

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _soft_min_score(points, beta=50.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp.
    Returns -soft_min(areas) / hull_area for minimization.
    """
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 1e10
        
    if hull_area < 1e-12:
        return 1e10
        
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft minimum via log-sum-exp: min(a) ≈ -(1/beta) * log(sum(exp(-beta*a)))
    # For numerical stability, subtract max first
    neg_areas = -areas
    max_neg = np.max(neg_areas)
    if max_neg < -1000:
        soft_min = np.exp(max_neg - 1/beta) * np.sum(np.exp(neg_areas - max_neg))
    else:
        soft_min = -(1.0/beta) * np.log(np.sum(np.exp(neg_areas - max_neg))) - max_neg/beta
        # Actually: soft_min = -(1/beta)*log(sum(exp(-beta*a)))
        # = -(1/beta)*log(sum(exp(neg_areas)))
        # = -(1/beta)*(log(sum(exp(neg_areas - max_neg))) + max_neg)
        # = -(1/beta)*log(sum(exp(neg_areas - max_neg))) - max_neg/beta
    
    return -soft_min / hull_area

def _generate_ring_points(params, n_out, n_in):
    """
    Generate points from ring parameters.
    params: [r, theta] for 2-ring
    """
    if len(params) == 2:
        r, theta = params
        angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
        outer_pts = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
        
        angles_in = theta + np.linspace(0, 2*np.pi, n_in, endpoint=False)
        inner_pts = r * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
        
        return np.vstack([outer_pts, inner_pts])
    else:
        raise ValueError("Only 2-param ring generation supported")

def _score_ring_2params(params, n_out, n_in):
    """
    Score for a 2-ring configuration defined by [r, theta].
    """
    try:
        points = _generate_ring_points(params, n_out, n_in)
        return _score(points)
    except Exception:
        return 0.0

def _soft_min_ring_score(params, n_out, n_in, beta=50.0):
    """
    Soft-min score for ring configuration (for minimization).
    """
    try:
        points = _generate_ring_points(params, n_out, n_in)
        return _soft_min_score(points, beta)
    except Exception:
        return 1e10

def _fast_local_score(points, moved_idx, old_point):
    """
    Fast score update when only one point moves.
    Only recompute triangles involving the moved point.
    Returns (new_min_area, hull_area) or None if hull fails.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return None
        
    if hull_area < 1e-12:
        return None
    
    # Get all triangle areas
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area, hull_area

def _coordinate_polish(points, max_iters=500, tol=1e-8, time_limit=10.0):
    """
    Coordinate descent polishing with fast local updates.
    Tries small perturbations in x and y for each point.
    """
    start_time = time.time()
    current_score = _score(points)
    if current_score <= 0:
        return points
    
    best_points = points.copy()
    best_score = current_score
    
    # Precompute which triples involve each point
    point_triples = [[] for _ in range(N)]
    for t_idx, (i, j, k) in enumerate(_TRIPLE_INDICES):
        point_triples[i].append(t_idx)
        point_triples[j].append(t_idx)
        point_triples[k].append(t_idx)
    
    for iteration in range(max_iters):
        if time.time() > start_time + time_limit:
            break
            
        improved = False
        
        for p_idx in range(N):
            if time.time() > start_time + time_limit:
                    break
                
            old_x, old_y = best_points[p_idx]
            
            # Try small steps in x and y
            for step in [1e-3, 1e-4, 1e-5, -1e-3, -1e-4, -1e-5]:
                for axis in [0, 1]:
                    candidate = best_points.copy()
                    if axis == 0:
                        candidate[p_idx, 0] = old_x + step
                    else:
                        candidate[p_idx, 1] = old_y + step
                    
                    # Quick check: is this point still in reasonable range?
                    if np.any(np.abs(candidate) > 2.0):
                        continue
                    
                    try:
                        hull = ConvexHull(candidate)
                        hull_area = hull.volume
                    except Exception:
                        continue
                        
                    if hull_area < 1e-12:
                        continue
                    
                    # Compute min triangle area
                    i_idx = _TRIPLE_INDICES[:, 0]
                    j_idx = _TRIPLE_INDICES[:, 1]
                    k_idx = _TRIPLE_INDICES[:, 2]
                    
                    p_i = candidate[i_idx]
                    p_j = candidate[j_idx]
                    p_k = candidate[k_idx]
                    
                    v1 = p_j - p_i
                    v2 = p_k - p_i
                    
                    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
                    areas = 0.5 * np.abs(cross)
                    
                    min_area = np.min(areas)
                    new_score = min_area / hull_area
                    
                    if new_score > best_score + tol:
                        best_score = new_score
                        best_points = candidate
                        improved = True
                        old_x, old_y = candidate[p_idx]
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_score = 0
    best_points = None
    
    # 1. Explore 2-ring configurations using Differential Evolution with soft-min
    splits = [(7, 7), (8, 6), (6, 8), (9, 5), (5, 9), (10, 4), (4, 10), (11, 3), (3, 11), (12, 2), (2, 12), (13, 1), (1, 13)]
    
    for n_out, n_in in splits:
        if time.time() > deadline - 10:
            break
            
        g = np.gcd(n_out, n_in)
        theta_max = 2 * np.pi / g
        bounds = [(0.0, 1.0), (0.0, theta_max)]
        
        def objective(p, _n_out=n_out, _n_in=n_in):
            return _soft_min_ring_score(p, _n_out, _n_in, beta=30.0)
        
        elapsed = time.time() - start_time
        remaining = deadline - 5 - elapsed
        
        if remaining < 2:
            break
            
        if remaining > 30:
            popsize = 15
            maxiter = 50
        elif remaining > 15:
            popsize = 10
            maxiter = 30
        else:
            popsize = 5
            maxiter = 10
            
        try:
            res = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-4,
                polish=True
            )
            
            if res.success:
                points = _generate_ring_points(res.x, n_out, n_in)
                score = _score(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    
        except Exception:
            pass
            
    # 2. Refine with Nelder-Mead on full 28D space using soft-min
    if best_points is not None and time.time() < deadline - 20:
        
        def objective_full(x):
            if time.time() > deadline - 3:
                raise TimeoutError()
            p = x.reshape(N, 2)
            return _soft_min_score(p, beta=50.0)
            
        try:
            res = minimize(
                objective_full, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9}
            )
            
            if res.fun < 1e10:
                new_pts = res.x.reshape(N, 2)
                s = _score(new_pts)
                if s > best_score:
                    best_score = s
                    best_points = new_pts
                    
        except TimeoutError:
            pass
        except Exception:
            pass

    # 3. Coordinate polish for final improvement
    if best_points is not None and time.time() < deadline - 5:
        polish_time = min(15.0, deadline - 5 - time.time())
        if polish_time > 1:
            polished = _coordinate_polish(best_points, max_iters=1000, tol=1e-9, time_limit=polish_time)
            polished_score = _score(polished)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished

    # 4. Fallback to regular polygon
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points

# EVOLVE-BLOCK-START
# The code above is the combined solution.
# It uses DE for global search over symmetric ring structures with soft-min surrogate,
# Nelder-Mead refinement with soft-min, and coordinate polish for final improvement.
# EVOLVE-BLOCK-END
