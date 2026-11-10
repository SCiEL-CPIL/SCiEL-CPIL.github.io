import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _hex_edge_normals(vertices):
    """Return unit edge normals for a hexagon (6, 2) array."""
    edges = np.roll(vertices, -1, axis=0) - vertices
    # inward normals: rotate edge by -90 degrees (for CCW vertices)
    normals = np.column_stack([-edges[:, 1], edges[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms
    # ensure they point inward: dot with (center - edge_midpoint) should be positive
    center = vertices.mean(axis=0)
    mids = (vertices + np.roll(vertices, -1, axis=0)) / 2
    dots = np.sum(normals * (center - mids), axis=1)
    normals[dots < 0] *= -1
    return normals

def _sat_overlap_penalty(v1, v2, normals1, normals2):
    """
    Compute separation along all 12 axes. Returns 0 if separated,
    penalty = (overlap_amount)^2 for each axis where they overlap.
    """
    all_normals = np.vstack([normals1, normals2])
    total_penalty = 0.0
    for n in all_normals:
        p1 = v1 @ n
        p2 = v2 @ n
        min1, max1 = p1.min(), p1.max()
        min2, max2 = p2.min(), p2.max()
        overlap = min(max1, max2) - max(min1, min2)
        if overlap > 0:
            total_penalty += overlap * overlap
    return total_penalty

def _point_outside_hex_penalty(point, outer_vertices):
    """
    For each outer edge (p1, p2), compute cross(p2-p1, point-p1).
    If any is negative, the point is outside. Penalty = sum of (negative values)^2.
    """
    penalty = 0.0
    for i in range(6):
        p1 = outer_vertices[i]
        p2 = outer_vertices[(i + 1) % 6]
        edge = p2 - p1
        to_point = point - p1
        cross_val = edge[0] * to_point[1] - edge[1] * to_point[0]
        if cross_val < 0:
            penalty += cross_val * cross_val
    return penalty

def _objective(params, angle_inner, angle_outer, outer_center):
    """
    params: 24 floats = 12 inner hex centers (x,y pairs)
    Returns total penalty + lambda * s (to minimize s implicitly via constraint)
    """
    centers = params.reshape(12, 2)
    penalty = 0.0
    
    # Precompute inner hex vertices and normals
    inner_verts = []
    inner_normals = []
    for i in range(12):
        cx, cy = centers[i]
        v = _hex_vertices(cx, cy, 1.0, angle_inner)
        inner_verts.append(v)
        inner_normals.append(_hex_edge_normals(v))
    
    # Pairwise overlap penalties
    for i in range(12):
        for j in range(i + 1, 12):
            # Quick reject: if centers are far apart, skip
            dist_sq = np.sum((centers[i] - centers[j]) ** 2)
            if dist_sq > 9.0:
                continue
            penalty += _sat_overlap_penalty(inner_verts[i], inner_verts[j], 
                                            inner_normals[i], inner_normals[j])
    
    # Calculate minimal s required to contain all inner hexagons
    # Assuming outer hex is centered at outer_center and has angle angle_outer
    # We want to minimize s.
    # s is determined by the max projection of inner vertices onto outer normals.
    
    # Outer normals (pointing outwards)
    a_outer = np.radians(angle_outer)
    # Normals are at angles a_outer + 30 + 60*k? 
    # Vertices are at a_outer + 60*k.
    # Midpoints of edges are at a_outer + 30 + 60*k.
    # Outward normals are at a_outer + 30 + 60*k.
    normal_angles = a_outer + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    max_proj = 0.0
    for v in inner_verts:
        projs = v @ normals.T
        # Distance from outer_center
        rel_v = v - outer_center
        rel_projs = rel_v @ normals.T
        max_proj = max(max_proj, np.max(rel_projs))
        
    # Apothem = s * sqrt(3) / 2
    # So s = 2 * max_proj / sqrt(3)
    s_required = 2.0 * max_proj / np.sqrt(3)
    
    # We want to minimize s_required, but also ensure no overlaps.
    # If we just minimize s_required, the hexagons will overlap.
    # So we add a large penalty for overlaps.
    total_cost = s_required + 100.0 * penalty
    
    return total_cost

def _is_feasible(params, s, angle_inner, angle_outer, outer_center, tol=1e-6):
    """Check if the configuration is feasible with given s."""
    centers = params.reshape(12, 2)
    
    # Precompute inner hex vertices and normals
    inner_verts = []
    inner_normals = []
    for i in range(12):
        cx, cy = centers[i]
        v = _hex_vertices(cx, cy, 1.0, angle_inner)
        inner_verts.append(v)
        inner_normals.append(_hex_edge_normals(v))
    
    # Pairwise separation check
    for i in range(12):
        for j in range(i + 1, 12):
            dist_sq = np.sum((centers[i] - centers[j]) ** 2)
            if dist_sq > 9.0:
                continue
            all_normals = np.vstack([inner_normals[i], inner_normals[j]])
            for n in all_normals:
                p1 = inner_verts[i] @ n
                p2 = inner_verts[j] @ n
                min1, max1 = p1.min(), p1.max()
                min2, max2 = p2.min(), p2.max()
                gap = min(max1, max2) - max(min1, min2)
                if gap < -tol:
                    return False
    
    # Boundary check
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], s, angle_outer)
    for i in range(12):
        for v in inner_verts[i]:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                edge = p2 - p1
                to_point = v - p1
                cross_val = edge[0] * to_point[1] - edge[1] * to_point[0]
                if cross_val < -tol:
                    return False
    
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = 620.0  # safety margin
    
    best_score = 0.0
    best_result = None
    
    # Try a specific good configuration: Hexagonal grid cluster
    # Angle 0 for all
    angle_inner = 0.0
    angle_outer = 0.0
    outer_center = np.array([0.0, 0.0])
    
    # Generate initial guess: 12 hexagons in a compact shape
    # Using axial coordinates (q, r)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    sqrt3 = np.sqrt(3)
    
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])

    # Shape: 
    # r=0: q=-1, 0, 1, 2 (4 hexes)
    # r=1: q=-2, -1, 0, 1 (4 hexes)
    # r=2: q=-1, 0 (2 hexes)
    # r=3: q=0 (1 hex)
    # Total 11. Need 1 more.
    
    # Let's try a symmetric "flower + 5"
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 corners: (2,0), (1,1), (0,2), (-1,1) is in R1... 
    # Let's just use a 3x4 grid roughly.
    
    initial_centers = []
    # Row 0 (y=0): 4 hexes
    for q in [-1, 0, 1, 2]:
        initial_centers.append(axial_to_cartesian(q, 0))
    # Row 1 (y=1.5): 4 hexes
    for q in [-2, -1, 0, 1]:
        initial_centers.append(axial_to_cartesian(q, 1))
    # Row 2 (y=3.0): 3 hexes
    for q in [-1, 0, 1]:
        initial_centers.append(axial_to_cartesian(q, 2))
    # Row 3 (y=4.5): 1 hex
    initial_centers.append(axial_to_cartesian(0, 3))
    
    initial_params = np.array(initial_centers).flatten()
    
    # Optimize
    try:
        result = minimize(_objective, initial_params, args=(angle_inner, angle_outer, outer_center),
                          method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
        
        if result.success or result.fun < 1e-6:
            # Extract s from the result
            centers = result.x.reshape(12, 2)
            # Recalculate s
            a_outer = np.radians(angle_outer)
            normal_angles = a_outer + np.radians(30 + 60 * np.arange(6))
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
            
            max_proj = 0.0
            for i in range(12):
                v = _hex_vertices(centers[i][0], centers[i][1], 1.0, angle_inner)
                rel_v = v - outer_center
                projs = rel_v @ normals.T
                max_proj = max(max_proj, np.max(projs))
            
            s = 2.0 * max_proj / np.sqrt(3)
            
            # Check feasibility
            if _is_feasible(result.x, s, angle_inner, angle_outer, outer_center):
                score = 1.0 / s
                if score > best_score:
                    best_score = score
                    inner_data = np.column_stack([centers, np.full(12, angle_inner)])
                    outer_data = np.array([outer_center[0], outer_center[1], angle_outer])
                    best_result = (inner_data, outer_data, s)
                    
    except Exception as e:
        pass
    
    # Fallback: Return a known valid solution if optimization failed
    if best_result is None:
        # Use the seed solution which is valid
        inner_hex_data = np.array(
            [
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
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        best_result = (inner_hex_data, outer_hex_data, outer_hex_side_length)
        
    return best_result

# EVOLVE-BLOCK-END
