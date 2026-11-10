# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = _START + 600.0


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(np.arange(0, 360, 60))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _edge_normals(vertices):
    # 6 outward edge normals for a CCW hexagon
    n = len(vertices)
    normals = np.empty((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        e = p2 - p1
        # outward normal (CCW polygon: right-hand normal of edge)
        normals[i] = np.array([e[1], -e[0]])
        L = np.linalg.norm(normals[i])
        if L > 0:
            normals[i] /= L
    return normals


def _min_gap_pair(v1, v2):
    # v1, v2: (6,2) vertex arrays of two unit hexagons
    # Compute separation on all 12 edge normals (6 from each)
    n1 = _edge_normals(v1)
    n2 = _edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    min_gap = np.inf
    for nrm in all_normals:
        p1_proj = v1 @ nrm
        p2_proj = v2 @ nrm
        # gap = max(0, (max_p1 - min_p2), (max_p2 - min_p1)) ... actually separation
        # If intervals overlap, gap = 0 (or negative if we want signed)
        # We want the minimum over normals of the separation distance
        # separation on this normal = max(min_p1 - max_p2, min_p2 - max_p1)
        sep = max(np.min(p1_proj) - np.max(p2_proj),
                  np.min(p2_proj) - np.max(p1_proj))
        if sep < min_gap:
            min_gap = sep
    return min_gap


def _outer_side_for_points(points, center):
    # points: (N, 2)
    # Find minimum side length s such that all points are inside regular hexagon
    # centered at `center` with some orientation. We'll try all 6 orientations? 
    # Actually, the outer hexagon orientation is also a variable in the full problem,
    # but for a given set of points and center, the min enclosing hexagon with
    # variable orientation can be found by checking 60-degree rotations? 
    # Simpler: fix outer hexagon orientation to 0 (flat-top or pointy-top) 
    # and just compute the required side. But the outer hexagon angle is a return value.
    # For optimization, we'll assume outer hexagon is axis-aligned (angle=0) 
    # and compute the minimal s. The returned angle can be 0.
    # Actually, to be safe, we should allow the outer hexagon to rotate.
    # But that adds 1 more variable. Let's include it in the optimization.
    # For now, compute min s for a given orientation.
    # We'll handle orientation in the main objective.
    pass


def _objective(params, n_inner=11):
    # params: [x1, y1, a1, x2, y2, a2, ..., x11, y11, a11, outer_s, outer_angle]
    # But we want to minimize outer_s, so it's not a parameter to optimize directly.
    # Instead, we compute the required outer_s from the inner positions.
    # Let's restructure: params = [x1, y1, a1, ..., x11, y11, a11]
    # and we compute the min enclosing hexagon side.
    
    # Extract inner hexagons
    inner = params.reshape(n_inner, 3)
    
    # Compute vertices for all inner hexagons
    verts = []
    for i in range(n_inner):
        cx, cy, a_deg = inner[i]
        v = _hex_vertices(cx, cy, 1.0, a_deg)
        verts.append(v)
    verts = np.array(verts)  # (11, 6, 2)
    
    # Penalty for overlaps
    overlap_penalty = 0.0
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            gap = _min_gap_pair(verts[i], verts[j])
            if gap < 1e-6:
                # Overlap or touching: penalty proportional to how much they overlap
                # gap is negative if overlapping? Let's check.
                # If intervals overlap, sep = max(min1-max2, min2-max1) which is negative.
                # So if gap < 1e-6, we have a problem.
                overlap_penalty += max(0, 1e-6 - gap) ** 2
    
    # Compute all vertices of all inner hexagons
    all_pts = verts.reshape(-1, 2)  # (66, 2)
    
    # Find center of mass
    center = np.mean(all_pts, axis=0)
    
    # Compute minimum enclosing regular hexagon side length
    # We'll try 6 orientations (0, 60, 120, ...) for the outer hexagon
    # and take the minimum side length.
    min_side = np.inf
    best_angle = 0.0
    for outer_angle_deg in range(0, 360, 60):
        outer_v = _hex_vertices(0, 0, 1.0, outer_angle_deg)  # unit hexagon
        outer_normals = _edge_normals(outer_v)
        
        # For each outer normal, project all points
        max_proj = -np.inf
        for nrm in outer_normals:
            projections = (all_pts - center) @ nrm
            max_proj = max(max_proj, np.max(projections))
        
        # The side length s must satisfy: for each normal, max_proj <= s * (distance from center to edge)
        # For a regular hexagon with side s, the distance from center to edge (apothem) is s * sqrt(3)/2
        # But the projection of a vertex on the normal direction... 
        # Actually, for a regular hexagon centered at origin with side s and orientation a,
        # the support function in direction n is s * max_i (v_i_unit @ n) where v_i are unit hex vertices.
        # So s >= max_proj / support_unit where support_unit = max_i (v_i_unit @ n)
        # But we need to check all 6 normals.
        # Let's compute the required s for each normal and take the max.
        required_s = 0.0
        for nrm in outer_normals:
            projections = (all_pts - center) @ nrm
            max_p = np.max(projections)
            # Unit hexagon support in direction nrm
            unit_support = np.max(outer_v @ nrm)
            if unit_support > 1e-12:
                s_needed = max_p / unit_support
            else:
                s_needed = 0
            required_s = max(required_s, s_needed)
        
        if required_s < min_side:
            min_side = required_s
            best_angle = float(outer_angle_deg)
    
    # If all points are at center, min_side could be 0
    if min_side < np.sqrt(11):
        min_side = np.sqrt(11)
    
    # Total objective: min_side + overlap_penalty
    total = min_side + 100.0 * overlap_penalty
    
    return total


def hexagon_packing_11():
    n_inner = 11
    
    # Initial guess: hexagonal cluster
    # Center at (0,0), 6 around at distance 2.5, 3 more at distance 5
    init = np.array([
        [0, 0, 0],
        [-2.5, 0, 0],
        [2.5, 0, 0],
        [-1.25, 2.17, 0],
        [1.25, 2.17, 0],
        [-1.25, -2.17, 0],
        [1.25, -2.17, 0],
        [-3.75, 2.17, 0],
        [3.75, 2.17, 0],
        [-3.75, -2.17, 0],
        [3.75, -2.17, 0],
    ]).flatten()
    
    best_params = init.copy()
    best_val = _objective(init, n_inner)
    
    # Optimization with deadline guard
    try:
        result = minimize(
            _objective,
            init,
            args=(n_inner,),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if result.fun < best_val:
            best_params = result.x
            best_val = result.fun
        
        # Check deadline
        if time.time() > _DEADLINE - 30:
            pass  # Skip further refinement
        
    except Exception:
        pass
    
    # Extract best parameters
    inner_hex_data = best_params.reshape(n_inner, 3)
    
    # Compute outer hexagon
    verts = []
    for i in range(n_inner):
        cx, cy, a_deg = inner_hex_data[i]
        v = _hex_vertices(cx, cy, 1.0, a_deg)
        verts.append(v)
    verts = np.array(verts)
    all_pts = verts.reshape(-1, 2)
    center = np.mean(all_pts, axis=0)
    
    # Find best outer orientation and side
    min_side = np.inf
    best_angle = 0.0
    for outer_angle_deg in range(0, 360, 60):
        outer_v = _hex_vertices(0, 0, 1.0, outer_angle_deg)
        outer_normals = _edge_normals(outer_v)
        
        required_s = 0.0
        for nrm in outer_normals:
            projections = (all_pts - center) @ nrm
            max_p = np.max(projections)
            unit_support = np.max(outer_v @ nrm)
            if unit_support > 1e-12:
                s_needed = max_p / unit_support
            else:
                s_needed = 0
            required_s = max(required_s, s_needed)
        
        if required_s < min_side:
            min_side = required_s
            best_angle = float(outer_angle_deg)
    
    if min_side < np.sqrt(11):
        min_side = float(np.sqrt(11))
    
    outer_hex_data = np.array([center[0], center[1], best_angle])
    outer_hex_side_length = float(min_side)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
