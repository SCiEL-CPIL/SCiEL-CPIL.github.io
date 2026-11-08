import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    if n < 3:
        return np.array([0.0])
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    """Compute area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _exact_score(points):
    """Calculate the exact score: min triangle area / hull area."""
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _soft_objective(params, beta=50.0):
    """Smoothed objective for L-BFGS-B. Minimizes negative soft-min area / hull area."""
    points = params.reshape(-1, 2)
    areas = _all_triangle_areas(points)
    # Soft min approximation
    scaled_areas = -beta * areas
    max_val = np.max(scaled_areas)
    if max_val == -np.inf:
        return 1e10
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled_areas - max_val)))
    soft_min = -log_sum_exp / beta
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 1e10
    return -soft_min / hull_area

def _refine_local(points, t0, deadline):
    """Refine a single point configuration using L-BFGS-B and Nelder-Mead."""
    best_points = points.copy()
    best_score = _exact_score(best_points)
    
    # Step 1: L-BFGS-B on soft objective
    # We use a higher beta for a sharper approximation of the min
    if time.time() - t0 < deadline - 5:
        try:
            x0 = best_points.flatten()
            # Use a moderate beta for smoothness
            res1 = minimize(_soft_objective, x0, args=(50.0,), method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            cand = res1.x.reshape(-1, 2)
            score1 = _exact_score(cand)
            if score1 > best_score:
                best_score = score1
                best_points = cand
        except Exception:
            pass

    # Step 2: Nelder-Mead on exact objective for final polish
    if time.time() - t0 < deadline - 2:
        try:
            x0 = best_points.flatten()
            res2 = minimize(lambda x: -_exact_score(x.reshape(-1, 2)), x0, method='Nelder-Mead',
                            options={'maxiter': 500, 'fatol': 1e-10, 'xatol': 1e-10, 'adaptive': True})
            cand = res2.x.reshape(-1, 2)
            score2 = _exact_score(cand)
            if score2 > best_score:
                best_score = score2
                best_points = cand
        except Exception:
            pass

    return best_points, best_score

def _generate_points(m_outer, k_inner, r_ratio, rot_angle):
    """Generate points: m_outer on unit circle, k_inner on circle of radius r_ratio, rotated by rot_angle."""
    points = []
    if m_outer > 0:
        angles_outer = 2 * np.pi * np.arange(m_outer) / m_outer
        for a in angles_outer:
            points.append([np.cos(a), np.sin(a)])
    if k_inner > 0:
        angles_inner = 2 * np.pi * np.arange(k_inner) / k_inner + rot_angle
        for a in angles_inner:
            points.append([r_ratio * np.cos(a), r_ratio * np.sin(a)])
    return np.array(points)

def _objective(params, m_outer, k_inner):
    """Negative score for minimization."""
    r_ratio, rot_angle = params
    # Clamp to valid range
    r_ratio = max(0.01, min(0.99, r_ratio))
    points = _generate_points(m_outer, k_inner, r_ratio, rot_angle)
    return -_exact_score(points)

def construct():
    n = N
    t0 = time.time()
    # Safety margin: The limit is 120s, we aim to finish by 110s
    deadline = t0 + 110.0
    
    best_points = None
    best_score = -np.inf
    
    # Strategy: 
    # 1. Explore structured multi-shell configurations (regular polygon + inner ring)
    # 2. Refine the best candidate using local optimization (L-BFGS-B + Nelder-Mead)
    
    # Partitions of 13 points into two shells (outer, inner)
    # We focus on balanced partitions which often yield better min-triangle-area
    partitions = [
        (7, 6), (8, 5), (6, 7), (5, 8), (9, 4), (4, 9), (10, 3), (3, 10), (11, 2), (2, 11)
    ]
    
    # Also try the single shell case (regular polygon) as a baseline
    partitions.append((13, 0))
    
    for m_outer, k_inner in partitions:
        if time.time() > deadline - 10:
            break
        
        if k_inner == 0:
            # Just a regular m-gon
            points = _generate_points(m_outer, 0, 0.0, 0.0)
            s = _exact_score(points)
            if s > best_score:
                best_score = s
                best_points = points
            continue
        
        # Multi-start Nelder-Mead optimization on the 2D parameter space (r_ratio, rot_angle)
        # Initial guesses for (r_ratio, rot_angle)
        # We try a few symmetric alignments and offsets
        initial_guesses = [
            (0.5, 0.0),
            (0.5, np.pi / k_inner),
            (0.3, 0.0),
            (0.7, 0.0),
            (0.5, np.pi / (2 * k_inner)),
            (0.4, np.pi / k_inner),
            (0.6, np.pi / k_inner),
            (0.5, 2 * np.pi / k_inner),
            (0.5, np.pi), # Anti-aligned
        ]
        
        for x0 in initial_guesses:
            if time.time() > deadline - 10:
                break
            
            try:
                result = minimize(
                    _objective,
                    x0,
                    args=(m_outer, k_inner),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                
                if result.success or result.fun < 0:
                    r_ratio, rot_angle = result.x
                    r_ratio = max(0.01, min(0.99, r_ratio))
                    points = _generate_points(m_outer, k_inner, r_ratio, rot_angle)
                    s = _exact_score(points)
                    if s > best_score:
                        best_score = s
                        best_points = points
            except Exception:
                pass
    
    # If we didn't find anything, fall back to regular 13-gon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _exact_score(best_points)

    # Refine the best candidate found using full 26D local optimization
    if time.time() < deadline - 5:
        refined_pts, refined_score = _refine_local(best_points, t0, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts

    return best_points
