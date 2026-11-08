import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

def _get_all_triangle_areas(points):
    """
    Calculate areas of all C(N,3) triangles formed by the points.
    Returns a numpy array of areas.
    """
    n = len(points)
    # Precompute indices for speed
    indices = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    i_idx, j_idx, k_idx = indices[:, 0], indices[:, 1], indices[:, 2]
    
    # Vectorized cross product calculation
    # Area = 0.5 * | (Pj - Pi) x (Pk - Pi) |
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross_z = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross_z)
    
    return areas

def _hull_area(points):
    """Calculate convex hull area."""
    try:
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
        
    areas = _get_all_triangle_areas(points)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _objective_smooth(points_flat, tau=100.0):
    """
    Smooth surrogate for the negative min triangle area.
    We want to maximize the minimum area, so we minimize -min_area.
    Since the hull area must be 1, we normalize inside.
    
    f(x) = min(area_i)
    smooth_f(x) = -1/tau * log(sum(exp(-tau * area_i)))
    
    We minimize -smooth_f(x) = 1/tau * log(sum(exp(-tau * area_i)))
    """
    points = points_flat.reshape(-1, 2)
    
    # 1. Center
    points = points - np.mean(points, axis=0)
    
    # 2. Normalize to unit hull area
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6  # Penalty for degenerate hull
    
    scale = 1.0 / np.sqrt(hull_area)
    points = points * scale
    
    # 3. Calculate areas
    areas = _get_all_triangle_areas(points)
    
    # 4. Smooth min approximation
    # To avoid overflow/underflow in exp:
    # log(sum(exp(-tau * areas))) = -tau * min_area + log(sum(exp(-tau * (areas - min_area))))
    m = np.min(areas)
    shifted = areas - m
    # shifted >= 0, so exp(-tau * shifted) <= 1, safe to sum
    sum_exp = np.sum(np.exp(-tau * shifted))
    
    # smooth_min = m - (1/tau) * log(sum_exp)
    # We want to minimize -smooth_min = (1/tau) * log(sum_exp) - m
    val = (1.0 / tau) * np.log(sum_exp) - m
    
    return val

def _objective_exact(points_flat):
    """
    Exact objective: -min_triangle_area, assuming points are normalized.
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
    
    # 3. Calculate min area
    areas = _get_all_triangle_areas(points)
    min_area = np.min(areas)
    
    return -min_area

def _generate_initial_candidates():
    """
    Generate diverse initial configurations for optimization.
    """
    candidates = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(pts.flatten())
    
    # 2. Multi-ring configurations
    # Try different ring distributions: e.g., 7 outer, 7 inner; 6-5-3; etc.
    ring_configs = [
        (7, 7, [1.0, 0.6]),
        (7, 7, [1.0, 0.5]),
        (5, 5, 4, [1.0, 0.7, 0.4]),
        (4, 5, 5, [1.0, 0.7, 0.4]),
        (3, 5, 6, [1.0, 0.7, 0.4]),
        (2, 6, 6, [1.0, 0.8, 0.5]),
        (1, 6, 7, [1.0, 0.8, 0.5]),
    ]
    
    for config in ring_configs:
        if len(config) == 3:
            n1, n2, radii = config
            n3 = 0
        else:
            n1, n2, n3, radii = config
            
        pts_list = []
        
        # Ring 1
        ang1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
        r1 = radii[0]
        pts1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
        pts_list.append(pts1)
        
        # Ring 2
        ang2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + (np.pi / n2 if n1 % 2 == 0 else np.pi / max(1, n1)) # Offset
        r2 = radii[1]
        pts2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
        pts_list.append(pts2)
        
        # Ring 3
        if n3 > 0:
            ang3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + (np.pi / n3 if n2 % 2 == 0 else np.pi / max(1, n2))
            r3 = radii[2]
            pts3 = np.column_stack([r3 * np.cos(ang3), r3 * np.sin(ang3)])
            pts_list.append(pts3)
        
        pts = np.vstack(pts_list)
        # Ensure exactly N points (pad or truncate if logic mismatch, though sums are 14)
        if len(pts) != N:
             # Fallback to regular polygon if config doesn't sum to 14
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
        candidates.append(pts.flatten())
        
    # 3. Random Gaussian
    np.random.seed(42)
    for _ in range(10):
        pts = np.random.randn(N, 2)
        candidates.append(pts.flatten())
        
    return candidates

def _perturb(x, step=0.1):
    """Perturb a point for basin hopping."""
    return x + np.random.randn(len(x)) * step

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin before 150s limit
    
    best_score = -1.0
    best_points = None
    
    # Get initial candidates
    inits = _generate_initial_candidates()
    
    # Optimization Strategy:
    # 1. For each candidate, run a local optimization using Powell on the smooth objective.
    # 2. Then polish with Nelder-Mead on the exact objective.
    
    for x0 in inits:
        if time.time() > deadline:
            break
        
        try:
            # Phase 1: Smooth local optimization
            # Powell is good for medium dimensions and doesn't need gradients
            res1 = minimize(_objective_smooth, x0, method='Powell', 
                            options={'maxiter': 300, 'xtol': 1e-6, 'ftol': 1e-6, 'adaptive': True})
            
            if time.time() > deadline:
                break
                
            p1 = res1.x.reshape(-1, 2)
            
            # Phase 2: Exact local optimization (Polish)
            res2 = minimize(_objective_exact, p1.flatten(), method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            if time.time() > deadline:
                break
                
            p2 = res2.x.reshape(-1, 2)
            
            # Normalize final result for consistency
            p2 = p2 - np.mean(p2, axis=0)
            hull_area = _hull_area(p2)
            if hull_area > 1e-10:
                p2 = p2 / np.sqrt(hull_area)
            
            s = _score(p2)
            
            if s > best_score:
                best_score = s
                best_points = p2
                
        except Exception as e:
            pass
    
    # If we have time, try a broader search using the best point found so far
    if time.time() < deadline - 30.0 and best_points is not None:
        # Manual Basin Hopping / Random Restarts around best
        for _ in range(15):
            if time.time() > deadline - 5.0:
                break
            noise = np.random.randn(N, 2) * 0.15
            p_perturbed = best_points + noise
            try:
                # Use a slightly more robust local optimizer
                res_refine = minimize(_objective_exact, p_perturbed.flatten(), method='Nelder-Mead', 
                                      options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                p_ref = res_refine.x.reshape(-1, 2)
                p_ref = p_ref - np.mean(p_ref, axis=0)
                hull_area = _hull_area(p_ref)
                if hull_area > 1e-10:
                    p_ref = p_ref / np.sqrt(hull_area)
                s = _score(p_ref)
                if s > best_score:
                    best_score = s
                    best_points = p_ref
            except Exception:
                pass

    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_points = best_points - np.mean(best_points, axis=0)
        hull_area = _hull_area(best_points)
        if hull_area > 1e-10:
             best_points = best_points / np.sqrt(hull_area)

    return best_points


