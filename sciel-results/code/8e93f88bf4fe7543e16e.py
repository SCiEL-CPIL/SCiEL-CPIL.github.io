"""Heilbronn convex regions: maximize min triangle area over convex hull area for 13 points."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import time

N = 13
_DEADLINE = 150.0
_START = time.time()
_SAFETY = 10.0

# Pre-compute triangle indices
TRI_INDICES = np.array(list(combinations(range(N), 3)))
NUM_TRI = len(TRI_INDICES)

def _min_triangle_area_exact(pts):
    """Minimum area of any triangle formed by 3 of the points. Vectorized."""
    p_i = pts[TRI_INDICES[:, 0]]
    p_j = pts[TRI_INDICES[:, 1]]
    p_k = pts[TRI_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)

def _convex_hull_area(pts):
    """Area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    return _min_triangle_area_exact(pts) / hull_area

def _log_sum_exp_surrogate(pts, temperature=0.01):
    """
    Smooth surrogate: -log(sum(exp(-area_i / T))) / N
    This approximates -min(area_i) when T is small, but is differentiable.
    We maximize this, so minimize its negative.
    """
    p_i = pts[TRI_INDICES[:, 0]]
    p_j = pts[TRI_INDICES[:, 1]]
    p_k = pts[TRI_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    # Use log-sum-exp for numerical stability
    max_a = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-(areas - max_a) / temperature))) + (-max_a / temperature)
    # We want to maximize min area, which is approximated by -log_sum * temperature
    # Actually, let's use a different smooth approximation:
    # min(a) ≈ -T * log(sum(exp(-a_i/T)))
    # So we want to maximize: -T * log(sum(exp(-a_i/T)))
    # Minimize: T * log(sum(exp(-a_i/T)))
    
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 1e10
    
    # The objective to minimize: T * log_sum_exp(-areas/T) / hull_area
    # This approximates min(areas)/hull_area when T is small
    surrogate_val = temperature * log_sum / hull_area
    return surrogate_val

def _neg_score_smooth(x, temperature=0.01):
    """Negative smooth score for minimization."""
    pts = x.reshape(N, 2)
    return _log_sum_exp_surrogate(pts, temperature)

def _generate_structured_inits():
    """Generate structured symmetric initial configurations."""
    inits = []
    
    # 1. Regular 13-gon centered at origin
    angles = 2 * np.pi * np.arange(N) / N
    pts_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts_reg)
    
    # 2. Regular 12-gon + center
    angles_12 = 2 * np.pi * np.arange(12) / 12
    pts_12 = np.column_stack([np.cos(angles_12), np.sin(angles_12)])
    pts_12_c = np.vstack([pts_12, np.array([0.0, 0.0])])
    inits.append(pts_12_c)
    
    # 3. Two concentric rings: 6 outer + 6 inner + 1 center
    angles_6o = 2 * np.pi * np.arange(6) / 6
    angles_6i = 2 * np.pi * np.arange(6) / 6 + np.pi / 6
    pts_6o = np.column_stack([np.cos(angles_6o), np.sin(angles_6o)])
    pts_6i = 0.5 * np.column_stack([np.cos(angles_6i), np.sin(angles_6i)])
    pts_rings = np.vstack([pts_6o, pts_6i, np.array([0.0, 0.0])])
    inits.append(pts_rings)
    
    # 4. Regular 13-gon with slight radial perturbation
    rng = np.random.RandomState(42)
    for trial in range(3):
        angles = 2 * np.pi * np.arange(N) / N
        radii = 1.0 + 0.1 * rng.randn(N)
        radii = np.clip(radii, 0.5, 1.5)
        pts_pert = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts_pert)
    
    # 5. Slightly rotated regular 13-gon
    for rot in [0.1, 0.2, 0.3]:
        angles = 2 * np.pi * np.arange(N) / N + rot
        pts_rot = np.column_stack([np.cos(angles), np.sin(angles)])
        inits.append(pts_rot)
    
    return inits

def construct():
    global _START, _DEADLINE
    elapsed = time.time() - _START
    remaining = _DEADLINE - _SAFETY - elapsed
    
    if remaining < 5.0:
        # Fallback: regular 13-gon
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_x = None
    best_score = -np.inf
    
    inits = _generate_structured_inits()
    
    # Evaluate initial points with exact score
    for init in inits:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_x = init.flatten()
    
    # Multi-start L-BFGS-B optimization using smooth surrogate
    # We use a decreasing temperature schedule to converge to the min-triangle objective
    
    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    
    for temp in temperatures:
        if time.time() - _START > _DEADLINE - _SAFETY - 3.0:
            break
        
        # Start from current best
        x_start = best_x.copy()
        
        try:
            res = minimize(
                _neg_score_smooth,
                x_start,
                args=(temp,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            
            if res.fun < 1e9:  # Valid result
                polished_x = res.x
                polished_score = _score(polished_x.reshape(N, 2))
                
                if polished_score > best_score:
                    best_score = polished_score
                    best_x = polished_x
                    
        except Exception:
            pass
    
    # Final exact polish using Nelder-Mead on the exact (non-smooth) objective
    if time.time() - _START < _DEADLINE - _SAFETY - 2.0:
        def _neg_exact(x):
            pts = x.reshape(N, 2)
            return -_score(pts)
        
        try:
            res = minimize(
                _neg_exact,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
            )
            polished_x = res.x
            polished_score = -res.fun
            
            if polished_score > best_score:
                best_score = polished_score
                best_x = polished_x
        except Exception:
            pass
    
    return best_x.reshape(N, 2)
