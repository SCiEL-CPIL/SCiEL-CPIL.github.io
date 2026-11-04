"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# Precompute triple indices for efficiency
_triple_indices = np.array(list(combinations(range(N), 3)))
_TRI_I = _triple_indices[:, 0]
_TRI_J = _triple_indices[:, 1]
_TRI_K = _triple_indices[:, 2]

# EVOLVE-BLOCK-START

def _compute_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        area = hull.volume
        if area < 1e-9:
            return 0.0
        return area
    except Exception:
        return 0.0

def _compute_min_ratio(pts):
    """
    Compute the minimum triangle area divided by the convex hull area.
    Returns the score (higher is better) or 0 if invalid.
    """
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 0.0
    
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

# Precompute geometry for support function parametrization
_ALPHAS = np.linspace(0, 2 * np.pi, N, endpoint=False)
_NORMALS = np.column_stack((np.cos(_ALPHAS), np.sin(_ALPHAS)))
# A_i maps [h_i, h_{i+1}] to P_i
# n_i . P_i = h_i
# n_{i+1} . P_i = h_{i+1}
# [n_i_x n_i_y] [x]   [h_i]
# [n_{i+1,x} n_{i+1,y}] [y] = [h_{i+1}]
_MATRICES = np.zeros((N, 2, 2))
_INV_MATRICES = np.zeros((N, 2, 2))
for i in range(N):
    _MATRICES[i, 0, :] = _NORMALS[i]
    _MATRICES[i, 1, :] = _NORMALS[(i + 1) % N]
    try:
        _INV_MATRICES[i] = np.linalg.inv(_MATRICES[i])
    except np.linalg.LinAlgError:
        _INV_MATRICES[i] = np.eye(2)

def _points_from_support(H):
    """
    Compute points from support function values H.
    P_i = A_i^{-1} [H_i, H_{i+1}]
    """
    H_ext = np.concatenate([H, H[:1]])
    pts = np.zeros((N, 2))
    for i in range(N):
        pts[i] = _INV_MATRICES[i] @ H_ext[i:i+2]
    return pts

def _score_from_H(H):
    """Compute score from support function parameters."""
    pts = _points_from_support(H)
    return _compute_min_ratio(pts)

def _objective_H(H):
    """Objective for minimization (minimize -score)."""
    return -_score_from_H(H)

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    # Phase 1: Differential Evolution on Support Function Parameters
    # Bounds for H: [0.5, 2.0] is a reasonable range for normalized shapes
    bounds = [(0.5, 2.0)] * N
    
    # Use scipy's differential evolution
    # maxiter=50, popsize=10 should be fast enough for 13 variables
    de_result = None
    try:
        from scipy.optimize import differential_evolution
        de_result = differential_evolution(
            _objective_H,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=12345,
            polish=False
        )
    except Exception:
        pass

    best_H = None
    best_score = -np.inf
    
    if de_result is not None and de_result.success:
        best_H = de_result.x
        best_score = _score_from_H(best_H)
    else:
        # Fallback: Regular polygon support function
        best_H = np.ones(N)
        best_score = _score_from_H(best_H)

    # Phase 2: Local Polish with Nelder-Mead on the exact objective
    # Nelder-Mead is robust for nonsmooth objectives
    if (time.time() - start_time) < time_limit - 10.0:
        try:
            res = minimize(
                _objective_H,
                best_H,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            score = _score_from_H(res.x)
            if score > best_score:
                best_score = score
                best_H = res.x
        except Exception:
            pass

    # Phase 3: Further polish with L-BFGS-B if we want smooth gradients? 
    # The objective is nonsmooth (min of areas), so Nelder-Mead is better.
    # However, we can try a few more Nelder-Mead restarts with perturbations.
    if (time.time() - start_time) < time_limit - 5.0:
        for _ in range(3):
            if (time.time() - start_time) > time_limit - 5.0:
                break
            try:
                perturbed_H = best_H + np.random.normal(0, 0.01, N)
                res = minimize(
                    _objective_H,
                    perturbed_H,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                score = _score_from_H(res.x)
                if score > best_score:
                    best_score = score
                    best_H = res.x
            except Exception:
                pass

    # Reconstruct final points
    final_pts = _points_from_support(best_H)
    
    return final_pts

# EVOLVE-BLOCK-END
