"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s


def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if area < min_area:
                    min_area = area
    if min_area == float('inf') or min_area <= 0:
        return 0.0
    return min_area / hull_area


def _make_two_layer_config(k, m, r_inner, theta_offset):
    """Create a two-layer regular polygon configuration.
    k points on outer unit circle, m points on inner circle of radius r_inner.
    theta_offset rotates the inner polygon relative to outer.
    """
    outer_angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    inner_angles = outer_angles + theta_offset  # shifted by theta_offset
    # Wait, inner has m points, not k. Let me fix.
    inner_angles = np.linspace(0, 2 * np.pi, m, endpoint=False) + theta_offset
    
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_pts = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts


def _optimize_partition(k, m, time_budget=2.0):
    """Optimize radius and rotation for a given (k, m) partition."""
    if time.time() - _START_TIME > _DEADLINE:
        return None, 0.0
    
    best_score = 0.0
    best_params = None
    
    # Try multiple initial guesses for (r_inner, theta_offset)
    r_guesses = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    theta_guesses = [0.0, np.pi / m, np.pi / (2 * m), np.pi / k, np.pi / (2 * k)]
    
    for r0 in r_guesses:
        if time.time() - _START_TIME > _DEADLINE:
            break
        for th0 in theta_guesses:
            if time.time() - _START_TIME > _DEADLINE:
                break
            
            def neg_score(params):
                r, th = params
                if r <= 0 or r >= 1:
                    return 1e10
                pts = _make_two_layer_config(k, m, r, th)
                s = _score_points(pts)
                return -s
            
            try:
                res = minimize(neg_score, [r0, th0], method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < 1e9:
                    s = -res.fun
                    if s > best_score:
                        best_score = s
                        best_params = res.x
            except Exception:
                pass
    
    return best_params, best_score


def construct():
    """Construct 13 points maximizing min triangle area / hull area."""
    best_score = 0.0
    best_pts = None
    
    # All integer partitions of 12 with k>=3, m>=3 (k+m=12)
    partitions = [(k, m) for k in range(3, 10) for m in range(3, 10) if k + m == 12]
    # That gives (3,9),(4,8),(5,7),(6,6),(7,5),(8,4),(9,3)
    
    # Also consider (k,m) where one layer has 1 or 2 points? 
    # But with 13 points, we need k+m=12 for two layers plus maybe 1 center?
    # Actually the task says 13 points total. Let me reconsider.
    # Two layers: k + m = 13? Or k + m + 1 (center) = 13?
    
    # Let me try k + m = 13 (two layers, no center)
    partitions_13 = [(k, m) for k in range(3, 11) for m in range(3, 11) if k + m == 13]
    # (3,10),(4,9),(5,8),(6,7),(7,6),(8,5),(9,4),(10,3)
    
    # Also try k + m + 1 = 13 (two layers + 1 center point)
    partitions_12 = [(k, m) for k in range(3, 10) for m in range(3, 10) if k + m == 12]
    # (3,9),(4,8),(5,7),(6,6),(7,5),(8,4),(9,3)
    
    all_partitions = partitions_13 + partitions_12
    
    # First, quickly evaluate all partitions with a few simple parameter choices
    for k, m in all_partitions:
        if time.time() - _START_TIME > _DEADLINE:
            break
        
        is_centered = (k + m == 12)  # has center point
        
        for r in [0.3, 0.5, 0.7]:
            for th in [0.0, np.pi / max(m, 1), np.pi / (2 * max(m, 1))]:
                if time.time() - _START_TIME > _DEADLINE:
                    break
                if is_centered:
                    outer_angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
                    inner_angles = np.linspace(0, 2 * np.pi, m, endpoint=False) + th
                    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
                    inner_pts = np.column_stack([r * np.cos(inner_angles), r * np.sin(inner_angles)])
                    center = np.array([[0.0, 0.0]])
                    pts = np.vstack([outer_pts, inner_pts, center])
                else:
                    pts = _make_two_layer_config(k, m, r, th)
                
                s = _score_points(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
    
    # Now optimize the top few partitions more carefully
    if time.time() - _START_TIME < _DEADLINE - 10:
        # Try optimizing a subset of promising partitions
        for k, m in [(5, 8), (6, 7), (4, 9), (7, 6), (5, 7), (6, 6)]:
            if time.time() - _START_TIME > _DEADLINE - 5:
                break
            is_centered = (k + m == 12)
            
            if is_centered:
                def neg_score_centered(params):
                    r, th = params
                    if r <= 0.05 or r >= 0.95:
                        return 1e10
                    outer_angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
                    inner_angles = np.linspace(0, 2 * np.pi, m, endpoint=False) + th
                    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
                    inner_pts = np.column_stack([r * np.cos(inner_angles), r * np.sin(inner_angles)])
                    center = np.array([[0.0, 0.0]])
                    pts = np.vstack([outer_pts, inner_pts, center])
                    s = _score_points(pts)
                    return -s
                
                for r0 in [0.4, 0.5, 0.6]:
                    for th0 in [0.0, np.pi / m]:
                        if time.time() - _START_TIME > _DEADLINE - 3:
                            break
                        try:
                            res = minimize(neg_score_centered, [r0, th0], method='Nelder-Mead',
                                           options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7})
                            if res.fun < 1e9:
                                s = -res.fun
                                if s > best_score:
                                    best_score = s
                                    r, th = res.x
                                    outer_angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
                                    inner_angles = np.linspace(0, 2 * np.pi, m, endpoint=False) + th
                                    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
                                    inner_pts = np.column_stack([r * np.cos(inner_angles), r * np.sin(inner_angles)])
                                    center = np.array([[0.0, 0.0]])
                                    best_pts = np.vstack([outer_pts, inner_pts, center])
                        except Exception:
                            pass
            else:
                def neg_score_no_center(params):
                    r, th = params
                    if r <= 0.05 or r >= 0.95:
                        return 1e10
                    pts = _make_two_layer_config(k, m, r, th)
                    s = _score_points(pts)
                    return -s
                
                for r0 in [0.4, 0.5, 0.6]:
                    for th0 in [0.0, np.pi / m]:
                        if time.time() - _START_TIME > _DEADLINE - 3:
                            break
                        try:
                            res = minimize(neg_score_no_center, [r0, th0], method='Nelder-Mead',
                                           options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7})
                            if res.fun < 1e9:
                                s = -res.fun
                                if s > best_score:
                                    best_score = s
                                    r, th = res.x
                                    best_pts = _make_two_layer_config(k, m, r, th)
                        except Exception:
                            pass
    
    if best_pts is None:
        # Fallback: 13 points on unit circle
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts
# EVOLVE-BLOCK-END
