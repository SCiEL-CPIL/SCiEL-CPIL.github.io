import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 11
D = math.sqrt(3)

def _hex_vertex_offsets(angle_deg, s=1.0):
    """Vertex offsets (relative to center) for a unit hexagon at given angle."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    return np.column_stack([s * np.cos(angles), s * np.sin(angles)])

def _outer_normals(angle_deg):
    """Outward unit normals for the 6 edges of a regular hexagon."""
    angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + angle_deg)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _flower_centers():
    """Centers of the 11 hexagons in the rigid flower pattern.
    1 at origin, 6 at distance sqrt(3), 4 at distance 3."""
    centers = np.zeros((11, 2))
    centers[0] = [0.0, 0.0]
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers[k+1] = [D * np.cos(angle), D * np.sin(angle)]
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers[k+7] = [3.0 * np.cos(angle), 3.0 * np.sin(angle)]
    return centers

def _compute_side(centers, inner_angle_deg, outer_angle_deg):
    """
    Compute minimum outer hexagon side length for given inner hex centers,
    inner hex rotation, and outer hex rotation.
    
    For a regular hexagon centered at origin with side S and rotation theta,
    a point p is inside iff for all 6 outward normals n_k: n_k . p <= S * sqrt(3)/2
    """
    normals = _outer_normals(outer_angle_deg)
    cos30 = math.sqrt(3) / 2.0
    offsets = _hex_vertex_offsets(inner_angle_deg)
    
    max_proj = 0.0
    for i in range(len(centers)):
        vertices = offsets + centers[i]
        projs = vertices @ normals.T
        max_proj = max(max_proj, np.max(projs))
    
    return max_proj / cos30

def _side_for_outer_rotation(theta_deg):
    """
    The flower is rigid. All inner hexagons share the same rotation.
    We fix inner rotation at 0 and vary outer hexagon rotation.
    Due to 6-fold symmetry, search over [0, 60).
    """
    centers = _flower_centers()
    return _compute_side(centers, 0.0, theta_deg)

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    centers = _flower_centers()
    
    best_side = float('inf')
    best_angle = 0.0

    # Coarse grid search over [0, 60)
    n_coarse = 1200
    angles = np.linspace(0, 60, n_coarse, endpoint=False)
    for a in angles:
        s = _side_for_outer_rotation(a)
        if s < best_side:
            best_side = s
            best_angle = a

    # Refine with bounded scalar minimization
    idx = int(np.argmin([_side_for_outer_rotation(a) for a in angles]))
    lo = angles[max(0, idx - 1)]
    hi = angles[min(n_coarse - 1, idx + 1)]
    if lo >= hi:
        lo = max(0.0, best_angle - 0.5)
        hi = min(60.0, best_angle + 0.5)

    try:
        res = minimize_scalar(_side_for_outer_rotation, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e-10, 'maxiter': 300})
        if res.success and res.fun < best_side:
            best_side = res.fun
            best_angle = res.x
    except Exception:
        pass

    # Ultra-fine refinement
    if time.time() < DEADLINE - 5:
        try:
            res2 = minimize_scalar(_side_for_outer_rotation,
                                   bounds=(best_angle - 0.001, best_angle + 0.001),
                                   method='bounded',
                                   options={'xatol': 1e-13, 'maxiter': 200})
            if res2.success and res2.fun < best_side:
                best_side = res2.fun
                best_angle = res2.x
        except Exception:
            pass

    # Build inner hexagon list: (x, y, angle_degrees)
    inner = []
    for i in range(11):
        inner.append([centers[i, 0], centers[i, 1], 0.0])

    # Tiny epsilon for strict containment
    best_side += 1e-9

    return inner, (0.0, 0.0), best_side, best_angle
