import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, a_deg):
    a_rad = np.deg2rad(a_deg)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_outer_side(positions, angle_deg):
    """
    Compute the minimum side length of a regular hexagon with the given angle
    that encloses all unit hexagons at the given positions.
    """
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    
    # All inner hexagon vertices
    # positions: (12, 2)
    # unit_hex_verts: (6, 2)
    all_verts = positions[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    
    a_rad = np.deg2rad(angle_deg)
    
    # Outward normals of the outer hexagon edges
    # Edge i connects vertex i and i+1.
    # The normal to the edge between vertex i and i+1 points outward.
    # Vertex i is at angle a + 60i. Vertex i+1 is at angle a + 60(i+1).
    # Midpoint angle is a + 60i + 30.
    normals = []
    for i in range(6):
        n_angle = a_rad + i * np.pi/3 + np.pi/6
        n_unit = np.array([np.cos(n_angle), np.sin(n_angle)])
        normals.append(n_unit)
    normals = np.array(normals) # (6, 2)
    
    # Project all vertices onto normals
    projections = all_verts_flat @ normals.T # (36, 6)
    max_projections = np.max(projections, axis=0) # (6,)
    
    # For a regular hexagon of side s, the distance from center to edge (apothem) is s * sqrt(3) / 2.
    # So max_proj <= s * sqrt(3) / 2
    # s >= 2 * max_proj / sqrt(3)
    min_s = np.max(max_projections) * 2 / np.sqrt(3)
    
    return min_s

def _check_validity(inner, outer, s):
    """
    Validates the solution according to the task contract.
    Returns True if valid, False otherwise.
    """
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    
    if s < np.sqrt(12) - 1e-9:
        return False
        
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False
                
    inner_verts_list = []
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1, ang)
        # Check unit edge length
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False
        inner_verts_list.append(verts)

    # Check separation (SAT)
    for i in range(12):
        for j in range(i+1, 12):
            va = inner_verts_list[i]
            vb = inner_verts_list[j]
            
            separated = False
            # Check axes from A
            for k in range(6):
                v1 = va[k]
                v2 = va[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if separated: continue
            
            # Check axes from B
            for k in range(6):
                v1 = vb[k]
                v2 = vb[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if not separated:
                return False
                
    # Check inner vertices inside outer
    for i in range(12):
        for v in inner_verts_list[i]:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                e = p2 - p1
                cross_val = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
                if cross_val < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
                
    return True

def _objective(x):
    """
    Objective function for optimization.
    x: array of length 25.
    x[0:24]: positions of 12 hexagons (12 * 2)
    x[24]: outer angle (radians)
    """
    positions = x[:24].reshape(12, 2)
    outer_angle_rad = x[24]
    outer_angle_deg = np.rad2deg(outer_angle_rad)
    
    s = _min_outer_side(positions, outer_angle_deg)
    
    # Penalty for overlaps
    # We use a simple distance-based penalty for speed during optimization.
    # If distance between centers < 1.8, they might overlap.
    # For unit hexagons, the minimum distance between centers for non-overlap is sqrt(3) ~ 1.732 if aligned, 
    # but can be less if rotated. However, 1.8 is a safe heuristic threshold for "close".
    # A better penalty is based on actual SAT or vertex distance, but that is expensive.
    # Let's use a softer penalty: sum of (threshold - dist)^2 for dist < threshold.
    penalty = 0.0
    threshold = 1.75 # Slightly less than 1.8 to allow tight packing
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < threshold:
                penalty += (threshold - d)**2
    
    # Penalty for containment (vertices outside outer hexagon)
    # This is handled by s being large enough, but if s is minimized, 
    # the hexagons might poke out if the angle is bad.
    # _min_outer_side already ensures containment by definition of s.
    # So the only real constraint is non-overlap.
    
    return s + penalty * 10.0

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580 # Safety margin
    
    best_score = 0.0
    best_inner = None
    best_outer = None
    best_s = None
    
    # Initial guess: Hexagonal lattice
    h = np.sqrt(3)/2
    init_pos = np.array([
        [0, 0],
        [2*h, 0],
        [h, 1.5],
        [-h, 1.5],
        [-2*h, 0],
        [-h, -1.5],
        [h, -1.5],
        [3*h, 1.5],
        [0, 3],
        [-3*h, 1.5],
        [-3*h, -1.5],
        [3*h, -1.5]
    ])
    
    # Multi-start Nelder-Mead
    num_starts = 5
    np.random.seed(42)
    
    for start_idx in range(num_starts):
        if time.time() > deadline:
            break
            
        # Perturb initial position
        noise_scale = 0.1 if start_idx > 0 else 0.0
        pos = init_pos + np.random.randn(12, 2) * noise_scale
        angle_rad = np.random.uniform(0, 2*np.pi) if start_idx > 0 else 0.0
        
        x0 = np.concatenate([pos.flatten(), [angle_rad]])
        
        # Run Nelder-Mead
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        except Exception:
            continue
            
        if time.time() > deadline:
            break
            
        # Extract result
        best_x = res.x
        positions = best_x[:24].reshape(12, 2)
        outer_angle_rad = best_x[24]
        outer_angle_deg = np.rad2deg(outer_angle_rad)
        
        s_candidate = _min_outer_side(positions, outer_angle_deg)
        
        # Construct inner and outer arrays
        inner_candidate = np.zeros((12, 3))
        inner_candidate[:, :2] = positions
        inner_candidate[:, 2] = 0.0 # Inner hexagons are aligned with 0 degrees
        
        outer_candidate = np.array([0, 0, outer_angle_deg])
        
        # Check validity
        if _check_validity(inner_candidate, outer_candidate, s_candidate):
            score_candidate = 1.0 / s_candidate
            if score_candidate > best_score:
                best_score = score_candidate
                best_inner = inner_candidate
                best_outer = outer_candidate
                best_s = s_candidate
                
    # Fallback if no valid solution found
    if best_inner is None:
        # Use the initial lattice as a fallback, ensuring it's valid
        # The initial lattice is likely valid if s is large enough
        # Let's just compute s for the initial lattice and return it
        # It might not be optimal, but it's a valid packing
        pos = init_pos
        s_candidate = _min_outer_side(pos, 0)
        
        inner_candidate = np.zeros((12, 3))
        inner_candidate[:, :2] = pos
        inner_candidate[:, 2] = 0.0
        
        outer_candidate = np.array([0, 0, 0])
        
        if _check_validity(inner_candidate, outer_candidate, s_candidate):
            best_inner = inner_candidate
            best_outer = outer_candidate
            best_s = s_candidate
        else:
            # This should not happen, but just in case, return a very loose packing
            inner_fallback = np.array([
                [0, 0, 0], [3, 0, 0], [-3, 0, 0],
                [1.5, 2.6, 0], [-1.5, 2.6, 0],
                [1.5, -2.6, 0], [-1.5, -2.6, 0],
                [4.5, 2.6, 0], [-4.5, 2.6, 0],
                [4.5, -2.6, 0], [-4.5, -2.6, 0],
                [0, 5.2, 0]
            ])
            outer_fallback = np.array([0, 0, 0])
            s_fallback = _min_outer_side(inner_fallback[:, :2], 0)
            return inner_fallback, outer_fallback, s_fallback
            
    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
