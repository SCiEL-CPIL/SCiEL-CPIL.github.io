# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _face_normals(a_deg):
    a = np.deg2rad(a_deg)
    mid_angles = a + np.deg2rad(30 + 60 * np.arange(6))
    return np.column_stack([np.cos(mid_angles), np.sin(mid_angles)])

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner, outer, side).
    """
    # 1. Define the 11 centers for the base cluster (Center + 6 Ring 1)
    centers = [[0.0, 0.0]]
    for k in range(6):
        ang = np.deg2rad(60 * k)
        centers.append([2.0 * np.cos(ang), 2.0 * np.sin(ang)])
    
    # 2. Define the 12 candidate positions for Ring 2
    candidates = []
    for k in range(6):
        ang = np.deg2rad(60 * k)
        candidates.append([4.0 * np.cos(ang), 4.0 * np.sin(ang)])
    for k in range(6):
        ang = np.deg2rad(60 * k + 30)
        candidates.append([2.0 * np.sqrt(3.0) * np.cos(ang), 2.0 * np.sqrt(3.0) * np.sin(ang)])
        
    candidates = np.array(candidates)
    
    best_side = np.inf
    best_combo = None
    best_orientation = 0.0
    best_centers = None
    best_centroid = None
    
    # 3. Iterate over all combinations of 4 from the 12 candidates
    for combo in combinations(range(12), 4):
        all_centers = np.array(centers)
        selected_centers = candidates[list(combo)]
        full_centers = np.vstack([all_centers, selected_centers])
        
        # Compute vertices for all 11 hexagons (angle 0)
        verts = np.empty((11, 6, 2))
        for i in range(11):
            verts[i] = _hex_vertices(full_centers[i, 0], full_centers[i, 1], 1.0, 0.0)
        
        all_verts = verts.reshape(-1, 2)
        centroid = all_verts.mean(axis=0)
        shifted_verts = all_verts - centroid
        
        # Optimize orientation of outer hexagon
        alphas = np.linspace(0, 60, 1000, endpoint=False)
        min_side_for_combo = np.inf
        best_alpha = 0.0
        
        for a_deg in alphas:
            normals = _face_normals(a_deg)
            dots = shifted_verts @ normals.T
            max_proj = np.max(dots)
            side = 2.0 * max_proj / np.sqrt(3.0)
            if side < min_side_for_combo:
                min_side_for_combo = side
                best_alpha = a_deg
        
        if min_side_for_combo < best_side:
            best_side = min_side_for_combo
            best_combo = combo
            best_orientation = best_alpha
            best_centers = full_centers
            best_centroid = centroid

    # 4. Apply scaling to ensure gap > 1e-6
    # The gap in the lattice is 0. Scaling centers by (1+eps) creates gap 2*eps.
    # We need gap > 1e-6, so 2*eps > 1e-6 => eps > 5e-7.
    # Use eps = 1e-5 for safety.
    eps = 1e-5
    scale_factor = 1.0 + eps
    
    # Scale the centers relative to the centroid
    # Note: The centroid of the vertices is the same as the centroid of the centers for a symmetric cluster?
    # Not exactly, but close. We should shift the scaled centers so the new centroid is at the origin?
    # The outer hexagon is centered at the origin.
    # We computed the optimal outer hexagon for the UNSCALED cluster centered at the origin.
    # If we scale the inner cluster, its "radius" increases by scale_factor.
    # So the new side length is best_side * scale_factor.
    
    new_side = best_side * scale_factor
    
    # Construct inner hexagons
    inner = np.zeros((11, 3))
    for i in range(11):
        # Shift by centroid, then scale
        cx = (best_centers[i, 0] - best_centroid[0]) * scale_factor
        cy = (best_centers[i, 1] - best_centroid[1]) * scale_factor
        inner[i, 0] = cx
        inner[i, 1] = cy
        inner[i, 2] = 0.0
        
    outer = np.array([0.0, 0.0, best_orientation])
    
    return inner, outer, float(new_side)

# EVOLVE-BLOCK-END
