import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 14

def _get_all_triangle_areas(points):
    """Calculate areas of all C(N,3) triangles formed by the points."""
    n = len(points)
    if n < 3:
        return np.array([])
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    
    v1 = points[j_idx] - points[i_idx]
    v2 = points[k_idx] - points[i_idx]
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return areas

def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 points."""
    areas = _get_all_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    return np.min(areas)

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        # Check for collinearity or degenerate cases
        if len(points) < 3:
            return 0.0
        # ConvexHull requires at least 3 non-collinear points
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
        
    min_area = _min_triangle_area(points)
    
    return min_area / hull_area

def _normalize_points(points):
    """Center and normalize to unit hull area."""
    points = points - np.mean(points, axis=0)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        # If degenerate, return as is or a small random jitter
        return points
    return points / np.sqrt(hull_area)

def _neg_objective(points_flat):
    """
    Objective function for minimization.
    Normalizes points to unit hull area internally to enforce constraint,
    then returns negative score.
    """
    points = points_flat.reshape(-1, 2)
    
    # 1. Center
    points = points - np.mean(points, axis=0)
    
    # 2. Normalize to unit hull area
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6
    
    scale = 1.0 / np.sqrt(hull_area)
    points = points * scale
    
    # 3. Calculate score
    min_area = _min_triangle_area(points)
    
    # 4. Return negative for minimization
    return -min_area

def _two_ring_points(params):
    """
    Two-ring parametrization: 7 outer, 7 inner points.
    params: [r_inner, angle_offset, *7 outer angle perturbations, *7 inner angle perturbations]
    Total 16 params.
    """
    r_inner = params[0]
    angle_offset = params[1]
    outer_pert = params[2:9]
    inner_pert = params[9:16]
    
    base_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer_angles = base_angles + outer_pert
    inner_angles = base_angles + angle_offset + inner_pert
    
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner = r_inner * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    points = np.vstack([outer, inner])
    return points

def _two_ring_objective(params):
    """
    Objective for two-ring optimization.
    """
    points = _two_ring_points(params)
    points = _normalize_points(points)
    min_area = _min_triangle_area(points)
    return -min_area

def _local_refine(points, max_iters=100, step_size=0.01):
    """
    Local refinement: Perturb points that are part of the worst triangles.
    Uses a directed search: try to move vertices of the smallest triangles apart.
    """
    points = points.copy()
    current_score = _score(points)
    if current_score == 0.0:
        return points, current_score
    
    n = len(points)
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    
    for _ in range(max_iters):
        areas = _get_all_triangle_areas(points)
        min_area = np.min(areas)
        
        # Find indices of triangles with area close to min
        # Using a tolerance to catch all "worst" triangles
        tol = 1e-5
        worst_tri_indices = np.where(np.abs(areas - min_area) < tol)[0]
        
        if len(worst_tri_indices) == 0:
            break
            
        # Get the set of points involved in the worst triangles
        worst_points = set()
        for tri_idx in worst_tri_indices[:20]: # Limit to 20 to keep it fast
            worst_points.add(i_idx[tri_idx])
            worst_points.add(j_idx[tri_idx])
            worst_points.add(k_idx[tri_idx])
        
        improved = False
        # Try to move each worst point
        for pt_idx in list(worst_points):
            # Generate candidate moves
            # 1. Small random perturbations
            # 2. Moves away from the centroid of the worst triangles involving this point
            
            # Calculate centroid of the "worst" triangles involving this point
            # This is a heuristic to push the point away from the cluster causing small areas
            centroids = []
            for tri_idx in worst_tri_indices[:20]:
                if i_idx[tri_idx] == pt_idx or j_idx[tri_idx] == pt_idx or k_idx[tri_idx] == pt_idx:
                    p1 = points[i_idx[tri_idx]]
                    p2 = points[j_idx[tri_idx]]
                    p3 = points[k_idx[tri_idx]]
                    c = (p1 + p2 + p3) / 3.0
                    centroids.append(c)
            
            if centroids:
                avg_centroid = np.mean(centroids, axis=0)
                direction = points[pt_idx] - avg_centroid
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    direction = direction / norm
                    # Try moving in the direction away from the centroid
                    for step in [step_size, step_size * 0.5, step_size * 2.0]:
                        new_points = points.copy()
                        new_points[pt_idx] = points[pt_idx] + direction * step
                        new_score = _score(new_points)
                        if new_score > current_score + 1e-12:
                            points = new_points
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                    
            # Also try small random perturbations
            for _ in range(5):
                perturb = np.random.randn(2) * step_size * 0.5
                new_points = points.copy()
                new_points[pt_idx] = points[pt_idx] + perturb
                new_score = _score(new_points)
                if new_score > current_score + 1e-12:
                    points = new_points
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
            
    return points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
    best_score = -1.0
    best_points = None
    
    def _update_best(points):
        nonlocal best_score, best_points
        if points is not None:
            s = _score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
    
    # Strategy 1: Two-Ring Differential Evolution
    # This is highly likely to contain the global optimum for N=14
    if time.time() < deadline - 30:
        try:
            # params: [r_inner, angle_offset, 7 outer perturbations, 7 inner perturbations]
            # r_inner is likely between 0.3 and 0.8
            # angle_offset is likely near pi/7 or 0
            bounds = [(0.2, 0.9), (-0.5, 0.5)] + [(-0.4, 0.4)] * 14
            
            res_de = differential_evolution(
                _two_ring_objective,
                bounds=bounds,
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-6,
                polish=True,
                init='sobol'
            )
            
            if time.time() < deadline - 10:
                p_two_ring = _two_ring_points(res_de.x)
                p_two_ring = _normalize_points(p_two_ring)
                _update_best(p_two_ring)
                
                # Polish with local refine
                if time.time() < deadline - 5:
                    p_refined, s_ref = _local_refine(p_two_ring, max_iters=50, step_size=0.005)
                    _update_best(p_refined)
                    
                    # Final Nelder-Mead polish
                    if time.time() < deadline - 2:
                        res_nm = minimize(_neg_objective, p_refined.flatten(), 
                                          method='Nelder-Mead', 
                                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                        p_polished = res_nm.x.reshape(-1, 2)
                        p_polished = _normalize_points(p_polished)
                        _update_best(p_polished)
        except Exception:
            pass
    
    # Strategy 2: General 2D Differential Evolution
    # Allows for non-symmetric solutions if they exist
    if time.time() < deadline - 20:
        try:
            # Initial guess: regular 14-gon
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            x0 = np.column_stack([np.cos(angles), np.sin(angles)])
            
            bounds = [(-2.0, 2.0)] * (2 * N)
            
            res_de2 = differential_evolution(
                _neg_objective,
                bounds=bounds,
                seed=123,
                maxiter=50, # Fewer iterations, DE is slow in 28D
                popsize=15,
                tol=1e-5,
                polish=False, # We will polish manually
                init='sobol'
            )
            
            if time.time() < deadline - 5:
                p_free = res_de2.x.reshape(-1, 2)
                p_free = _normalize_points(p_free)
                _update_best(p_free)
                
                if time.time() < deadline - 3:
                    p_refined, s_ref = _local_refine(p_free, max_iters=30, step_size=0.005)
                    _update_best(p_refined)
        except Exception:
            pass
    
    # Strategy 3: Multi-start Local Search from Heuristic Initializations
    # Try several specific topologies
    if time.time() < deadline - 5:
        inits = []
        
        # 1. Regular 14-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
        
        # 2. Two-ring 7/7 (0.5 radius)
        ang1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        outer = np.column_stack([np.cos(ang1), np.sin(ang1)])
        ang2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi/7
        inner = 0.5 * np.column_stack([np.cos(ang2), np.sin(ang2)])
        inits.append(np.vstack([outer, inner]))
        
        # 3. Two-ring 7/7 (0.6 radius)
        inner2 = 0.6 * np.column_stack([np.cos(ang2), np.sin(ang2)])
        inits.append(np.vstack([outer, inner2]))
        
        # 4. Two-ring 7/7 (0.4 radius)
        inner3 = 0.4 * np.column_stack([np.cos(ang2), np.sin(ang2)])
        inits.append(np.vstack([outer, inner3]))
        
        # 5. Random
        inits.append(np.random.randn(N, 2))
        
        for p_init in inits:
            if time.time() > deadline - 2:
                break
            try:
                p_norm = _normalize_points(p_init)
                p_refined, s_ref = _local_refine(p_norm, max_iters=40, step_size=0.005)
                _update_best(p_refined)
            except Exception:
                pass

    if best_points is None:
        # Fallback
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_points = _normalize_points(best_points)

    return best_points
