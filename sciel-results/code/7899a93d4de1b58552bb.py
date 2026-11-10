# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _face_normals(a_deg):
    a = np.deg2rad(a_deg)
    mid_angles = a + np.deg2rad(30 + 60 * np.arange(6))
    return np.column_stack([np.cos(mid_angles), np.sin(mid_angles)])

def _pair_gap(verts_a, verts_b, norms_a, norms_b):
    """
    Check separation using the 12 edge normals (6 from each hexagon).
    Returns the maximum gap over all 12 normals.
    If > 0, the hexagons are separated by at least that amount on some normal.
    """
    # Project onto normals of A
    proj_a = verts_a @ norms_a.T  # (6, 6)
    proj_b = verts_b @ norms_a.T  # (6, 6)
    gap_a = np.maximum(
        proj_b.min(axis=0) - proj_a.max(axis=0),
        proj_a.min(axis=0) - proj_b.max(axis=0)
    )
    
    # Project onto normals of B
    proj_a2 = verts_a @ norms_b.T  # (6, 6)
    proj_b2 = verts_b @ norms_b.T  # (6, 6)
    gap_b = np.maximum(
        proj_b2.min(axis=0) - proj_a2.max(axis=0),
        proj_a2.min(axis=0) - proj_b2.max(axis=0)
    )
    
    return max(gap_a.max(), gap_b.max())

def _objective(params, time_limit=600.0, start_time=0.0):
    """
    Objective function for optimization.
    Returns the side length of the minimal enclosing hexagon if valid,
    or a large penalty if invalid.
    """
    if time.time() - start_time > time_limit:
        return 1e10
        
    verts = np.empty((11, 6, 2))
    norms = np.empty((11, 6, 2))
    
    for i in range(11):
        cx, cy, ad = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        verts[i] = _hex_vertices(cx, cy, 1.0, ad)
        norms[i] = _face_normals(ad)
    
    # Check pairwise separation
    min_gap = np.inf
    for i in range(11):
        for j in range(i + 1, 11):
            g = _pair_gap(verts[i], verts[j], norms[i], norms[j])
            if g < min_gap:
                min_gap = g
            if min_gap < -1.0:
                # Heavy overlap, early exit
                return 1e10
    
    if min_gap < 1e-6:
        # Overlap penalty
        return 1e6 * (1e-6 - min_gap) + 100.0
    
    # Compute minimal enclosing regular hexagon
    all_verts = verts.reshape(-1, 2)
    centroid = all_verts.mean(axis=0)
    shifted = all_verts - centroid
    
    # Optimize orientation in [0, 60)
    best_side = np.inf
    for a_deg in np.linspace(0, 60, 121):
        n = _face_normals(a_deg)
        dots = shifted @ n.T  # (66, 6)
        r = dots.max()
        side = 2.0 * r / np.sqrt(3.0)
        if side < best_side:
            best_side = side
    
    return best_side

def _generate_grid_packing():
    """
    Generate a dense packing of 11 hexagons based on a hexagonal grid.
    This serves as a robust initial guess.
    """
    # Hex grid coordinates for a "side 3" hexagon (19 hexagons)
    # q, r are axial coordinates
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Check if within hexagon of radius 2
            if abs(q) <= 2 and abs(r) <= 2 and abs(q + r) <= 2:
                # Convert axial to Cartesian
                # Unit hexagon side 1, distance between centers 2
                x = 2.0 * (q + r / 2.0)
                y = 2.0 * (np.sqrt(3) / 2.0) * r
                centers.append((x, y))
    
    # Sort by distance to origin
    centers.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the 11 closest
    selected = centers[:11]
    
    params = []
    for x, y in selected:
        params.append([x, y, 0.0])
        
    return np.array(params).flatten()

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner, outer, side).
    """
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0
    
    # Initial guess: dense grid packing
    init_params = _generate_grid_packing()
    
    best_params = init_params.copy()
    best_side = _objective(best_params, time_limit, start_time)
    
    # Local refinement
    if time.time() - start_time < time_limit - safety_margin:
        remaining = time_limit - safety_margin - (time.time() - start_time)
        
        def obj(x):
            if time.time() - start_time > time_limit - safety_margin:
                return 1e10
            return _objective(x, time_limit, start_time)
        
        try:
            # Use Nelder-Mead for robustness without gradients
            res = minimize(
                obj, best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'fatol': 1e-10, 'xatol': 1e-10, 'adaptive': True}
            )
            if res.fun < best_side:
                best_params = res.x
                best_side = res.fun
        except Exception:
            pass
    
    # Build result
    inner_hex_data = np.empty((11, 3))
    for i in range(11):
        inner_hex_data[i] = best_params[3 * i:3 * i + 3]
    
    # Compute outer hexagon
    verts = np.empty((11, 6, 2))
    for i in range(11):
        cx, cy, ad = best_params[3 * i], best_params[3 * i + 1], best_params[3 * i + 2]
        verts[i] = _hex_vertices(cx, cy, 1.0, ad)
    
    all_verts = verts.reshape(-1, 2)
    centroid = all_verts.mean(axis=0)
    shifted = all_verts - centroid
    
    best_a = 0.0
    best_side_final = np.inf
    for a_deg in np.linspace(0, 60, 241):
        n = _face_normals(a_deg)
        dots = shifted @ n.T
        r = dots.max()
        side = 2.0 * r / np.sqrt(3.0)
        if side < best_side_final:
            best_side_final = side
            best_a = a_deg
    
    # Shift inner hexagons so centroid is at origin
    inner_hex_data[:, 0] -= centroid[0]
    inner_hex_data[:, 1] -= centroid[1]
    
    outer_hex_data = np.array([0.0, 0.0, best_a])
    outer_hex_side_length = float(best_side_final)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
