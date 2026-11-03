"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _triangle_areas(pts):
    """Compute all triangle areas among triples of points."""
    n = len(pts)
    if n < 3:
        return np.array([])
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                cross = np.abs((pts[j][0]-pts[i][0])*(pts[k][1]-pts[i][1]) - 
                              (pts[j][1]-pts[i][1])*(pts[k][0]-pts[i][0]))
                areas.append(0.5 * cross)
    return np.array(areas)

def _min_triangle_area(pts):
    areas = _triangle_areas(pts)
    return np.min(areas) if len(areas) > 0 else 0.0

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score_points(pts):
    if len(pts) < 3:
        return 0.0
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_area

def _neg_score(pts_flat):
    pts = pts_flat.reshape(-1, 2)
    return -_score_points(pts)

def _penalty_objective(pts_flat, target_t, penalty_weight):
    """
    Penalty-based objective for max-min optimization.
    Maximize min triangle area subject to hull area = 1.
    """
    pts = pts_flat.reshape(-1, 2)
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 1e10
    
    # Normalize to unit area
    scale = 1.0 / np.sqrt(hull_area)
    pts_norm = pts * scale
    
    areas = _triangle_areas(pts_norm)
    min_area = np.min(areas)
    
    # Penalty for triangles below target
    violations = np.maximum(0, target_t - areas)
    penalty = penalty_weight * np.sum(violations**2)
    
    # We want to maximize min_area, so minimize negative min_area + penalty
    return -min_area + penalty

def _refine_with_penalty(pts_init, target_t, penalty_weight, maxiter=500):
    """Refine using penalty-based optimization."""
    try:
        result = minimize(
            _penalty_objective,
            pts_init.flatten(),
            args=(target_t, penalty_weight),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if result.fun is not None:
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    return pts_init

def _local_refine_aggressive(pts, deadline):
    """Aggressive local refinement with multiple methods and restarts."""
    best_score = _score_points(pts)
    best_pts = pts.copy()
    
    # Method 1: Nelder-Mead on raw score
    try:
        if time.time() < deadline - 5:
            result = minimize(
                _neg_score,
                pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9}
            )
            if result.fun is not None:
                candidate = result.x.reshape(-1, 2)
                score = _score_points(candidate)
                if score > best_score:
                    best_score = score
                    best_pts = candidate
    except Exception:
        pass
    
    # Method 2: Powell method
    try:
        if time.time() < deadline - 5:
            result = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 200, 'ftol': 1e-9, 'xtol': 1e-9}
            )
            if result.fun is not None:
                candidate = result.x.reshape(-1, 2)
                score = _score_points(candidate)
                if score > best_score:
                    best_score = score
                    best_pts = candidate
    except Exception:
        pass
    
    # Method 3: Penalty-based refinement with increasing target
    try:
        if time.time() < deadline - 10:
            current_min = _min_triangle_area(best_pts)
            hull_area = _hull_area(best_pts)
            if hull_area > 1e-12:
                scale = 1.0 / np.sqrt(hull_area)
                norm_min = current_min * scale**2
                
                # Try penalty refinement with target slightly above current
                target = norm_min * 1.05
                refined = _refine_with_penalty(best_pts, target, 100.0, maxiter=200)
                score = _score_points(refined)
                if score > best_score:
                    best_score = score
                    best_pts = refined
    except Exception:
        pass
    
    # Random restarts from best point
    try:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            perturbation = np.random.normal(0, 0.03, size=best_pts.shape)
            perturbed = best_pts + perturbation
            result = minimize(
                _neg_score,
                perturbed.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 150, 'xatol': 1e-7, 'fatol': 1e-9}
            )
            if result.fun is not None:
                candidate = result.x.reshape(-1, 2)
                score = _score_points(candidate)
                if score > best_score:
                    best_score = score
                    best_pts = candidate
    except Exception:
        pass
    
    return best_pts

def _generate_heilbronn_init():
    """
    Generate initial configurations based on known Heilbronn problem structures.
    For n=13, known good configurations involve points on concentric circles
    or specific polygonal arrangements.
    """
    inits = []
    
    # Configuration 1: Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts.copy())
    
    # Configuration 2: 12-gon + center
    angles12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts12 = np.column_stack([np.cos(angles12), np.sin(angles12)])
    center = np.array([[0.0, 0.0]])
    inits.append(np.vstack([pts12, center]).copy())
    
    # Configuration 3: Two concentric rings (7+6)
    angles7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    pts7 = np.column_stack([np.cos(angles7), np.sin(angles7)])
    angles6 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    pts6 = 0.6 * np.column_stack([np.cos(angles6), np.sin(angles6)])
    inits.append(np.vstack([pts7, pts6]).copy())
    
    # Configuration 4: Three rings (5+4+4)
    angles5 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    pts5 = np.column_stack([np.cos(angles5), np.sin(angles5)])
    angles4a = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
    pts4a = 0.65 * np.column_stack([np.cos(angles4a), np.sin(angles4a)])
    angles4b = np.linspace(0, 2*np.pi, 4, endpoint=False)
    pts4b = 0.35 * np.column_stack([np.cos(angles4b), np.sin(angles4b)])
    inits.append(np.vstack([pts5, pts4a, pts4b]).copy())
    
    # Configuration 5: 9+4 rings
    angles9 = np.linspace(0, 2*np.pi, 9, endpoint=False)
    pts9 = np.column_stack([np.cos(angles9), np.sin(angles9)])
    angles4 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
    pts4 = 0.55 * np.column_stack([np.cos(angles4), np.sin(angles4)])
    inits.append(np.vstack([pts9, pts4]).copy())
    
    # Configuration 6: 8+5 rings
    angles8 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    pts8 = np.column_stack([np.cos(angles8), np.sin(angles8)])
    angles5 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
    pts5 = 0.6 * np.column_stack([np.cos(angles5), np.sin(angles5)])
    inits.append(np.vstack([pts8, pts5]).copy())
    
    # Configuration 7: 10+3 rings
    angles10 = np.linspace(0, 2*np.pi, 10, endpoint=False)
    pts10 = np.column_stack([np.cos(angles10), np.sin(angles10)])
    angles3 = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
    pts3 = 0.5 * np.column_stack([np.cos(angles3), np.sin(angles3)])
    inits.append(np.vstack([pts10, pts3]).copy())
    
    # Configuration 8: 11+2 rings
    angles11 = np.linspace(0, 2*np.pi, 11, endpoint=False)
    pts11 = np.column_stack([np.cos(angles11), np.sin(angles11)])
    angles2 = np.array([0, np.pi])
    pts2 = 0.45 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    inits.append(np.vstack([pts11, pts2]).copy())
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Generate diverse initializations
    inits = _generate_heilbronn_init()
    
    # Also add some random initializations
    for _ in range(5):
        pts = np.random.uniform(-1, 1, size=(N, 2))
        inits.append(pts)
    
    best_score = -1.0
    best_pts = None
    
    # Evaluate all initializations and pick the best
    for pts in inits:
        if time.time() > deadline - 30:
            break
        score = _score_points(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # If no valid points found, return first init
    if best_pts is None:
        return inits[0]
    
    # Aggressive local refinement
    if time.time() < deadline - 15:
        refined = _local_refine_aggressive(best_pts, deadline)
        score = _score_points(refined)
        if score > best_score:
            best_score = score
            best_pts = refined
    
    # Try refining other promising inits if time permits
    for pts in inits[:5]:
        if time.time() > deadline - 10:
            break
        if np.array_equal(pts, best_pts):
            continue
        score = _score_points(pts)
        if score > best_score * 0.95:  # Only refine if close to best
            refined = _local_refine_aggressive(pts, deadline)
            score = _score_points(refined)
            if score > best_score:
                best_score = score
                best_pts = refined
    
    return best_pts
# EVOLVE-BLOCK-END
