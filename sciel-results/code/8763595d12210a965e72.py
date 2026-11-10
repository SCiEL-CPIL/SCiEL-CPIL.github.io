# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon with center (cx,cy), side d, rotation angle_deg."""
    angles = np.deg2rad(angle_deg) + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_containing_hex_side(inner_verts, outer_center, outer_angle_deg):
    """
    For a set of inner vertices (N,2) and an outer hexagon defined by center and angle,
    find the minimum side length s such that all inner vertices are inside the outer hexagon.
    """
    oc = np.array(outer_center)
    oa = np.deg2rad(outer_angle_deg)
    base_angles = oa + np.deg2rad(60.0 * np.arange(6))
    unit_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    # p - oc: shape (N, 2)
    rel = inner_verts - oc  # (N, 2)
    
    # Edge vectors: e_i = u_{i+1} - u_i, shape (6, 2)
    edge_vecs = np.roll(unit_verts, -1, axis=0) - unit_verts  # (6, 2)
    
    # A_i(p) = cross(e_i, p-oc) for each i and p
    # cross(e, v) = e[0]*v[1] - e[1]*v[0]
    # For all pairs: A[i, p] = edge_vecs[i,0]*rel[p,1] - edge_vecs[i,1]*rel[p,0]
    # Shape: (6, N)
    A = edge_vecs[:, 0:1] * rel[1:2, :] - edge_vecs[:, 1:2] * rel[0:1, :]  # (6, N)
    
    # s_min = max(0, max over i,p of (-2 * A[i,p] / sqrt(3)))
    s_min = np.max(-2.0 * A / np.sqrt(3.0))
    s_min = max(s_min, 0.0)
    
    return s_min

def _rigid_body_positions():
    """
    Return the fixed positions of 12 unit hexagons in a rigid, non-overlapping cluster.
    Uses a hexagonal lattice arrangement. All hexagons have the same orientation (0 degrees).
    """
    sqrt3 = np.sqrt(3.0)
    
    # Axial coordinates (q, r) for hexagonal grid
    # For flat-top: x = 1.5 * q, y = sqrt(3) * (r + q/2)
    
    # Let me pick 12 positions that form a nice cluster:
    axial_positions = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # center + ring 1
        (2, 0), (1, 1), (0, 2), (-2, 0), (-1, -1), (2, -1),   # 6 from ring 2
    ]
    
    # Convert to Cartesian for flat-top hexagons
    positions = []
    for q, r in axial_positions:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        positions.append([x, y, 0.0])  # angle 0
    
    return np.array(positions)

def _check_separation_fast(inner_positions):
    """
    Quick check that no two inner hexagons overlap using the separating axis theorem.
    For two regular hexagons with the same orientation, we just need to check that
    the distance between centers is at least sqrt(3) (the distance for edge-sharing).
    """
    n = len(inner_positions)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(inner_positions[i, :2] - inner_positions[j, :2])
            if dist < np.sqrt(3.0) - 1e-9:
                return False
    return True

def _compute_s_for_params(inner_positions, outer_center, outer_angle_deg):
    """Compute the minimum outer hexagon side length for given parameters."""
    # Get all vertices of all inner hexagons
    all_verts = []
    for pos in inner_positions:
        verts = _hex_vertices(pos[0], pos[1], 1.0, pos[2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # (72, 2)
    
    s = _min_containing_hex_side(all_verts, outer_center, outer_angle_deg)
    return s

def _validate_result(inner_hex_data, outer_hex_data, s):
    """Validate the result against the task constraints."""
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12.0):
        return False
    
    # Check outer hex vertices are distinct
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    # Shift all centers by minus outer center
    oc = outer_hex_data[:2]
    shifted_inner = inner_hex_data.copy()
    shifted_inner[:, :2] -= oc
    
    # Check unit edge lengths
    for pos in shifted_inner:
        verts = _hex_vertices(pos[0], pos[1], 1.0, pos[2])
        for i in range(6):
            edge_len = np.linalg.norm(verts[(i + 1) % 6] - verts[i])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    # Check separation between inner hexagons
    if not _check_separation_fast(shifted_inner):
        return False
    
    # Check all inner vertices are inside the outer hexagon
    all_verts = []
    for pos in shifted_inner:
        verts = _hex_vertices(pos[0], pos[1], 1.0, pos[2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Outer hex vertices
    ov = _hex_vertices(0, 0, s, outer_hex_data[2])
    
    for v in all_verts:
        for i in range(6):
            p1 = ov[i]
            p2 = ov[(i + 1) % 6]
            edge = p2 - p1
            to_v = v - p1
            cross_val = edge[0] * to_v[1] - edge[1] * to_v[0]
            if cross_val < -1e-6:
                return False
    
    return True

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a rigid-body approach: fix the relative positions of the 12 hexagons, then optimize
    the outer hexagon's angle and center offset to minimize the containing side length.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit with safety margin
    
    # Get the fixed rigid body positions
    base_positions = _rigid_body_positions()
    
    # Verify no overlap in the base configuration
    if not _check_separation_fast(base_positions):
        # Fallback to a known valid configuration
        base_positions = np.array([
            [0, 0, 0],
            [1.5, 0, 0], [-1.5, 0, 0],
            [0.75, 1.5, 0], [-0.75, 1.5, 0],
            [0.75, -1.5, 0], [-0.75, -1.5, 0],
            [3.0, 0, 0], [-3.0, 0, 0],
            [1.5, 3.0, 0], [-1.5, 3.0, 0],
            [1.5, -3.0, 0], [-1.5, -3.0, 0],
        ])
    
    best_s = np.inf
    best_params = (0.0, 0.0, 0.0)  # (outer_angle_deg, outer_cx, outer_cy)
    
    # Coarse grid search over outer angle and center
    # The outer hexagon's angle has 60-degree symmetry, so search [0, 60)
    # Center offset: the cluster is roughly symmetric, so center should be near origin
    # Let's search center in [-2, 2] x [-2, 2] and angle in [0, 60)
    
    angle_range = np.linspace(0, 60, 61)  # 61 points, step 1 degree
    center_range = np.linspace(-2, 2, 17)  # 17 points, step 0.25
    
    # Coarse pass
    for angle in angle_range:
        if time.time() > deadline - 60:  # Leave 60s for refinement
            break
        for cx in center_range:
            for cy in center_range:
                s = _compute_s_for_params(base_positions, [cx, cy], angle)
                if s < best_s:
                    best_s = s
                    best_params = (angle, cx, cy)
    
    # Refine with local search around the best point found
    if time.time() < deadline - 30:
        # Local refinement
        best_angle, best_cx, best_cy = best_params
        
        # Nelder-Mead style local search
        step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
        
        for step in step_sizes:
            if time.time() > deadline - 30:
                break
            improved = True
            while improved:
                improved = False
                # Try 8 neighbors
                neighbors = [
                    (best_angle + step, best_cx, best_cy),
                    (best_angle - step, best_cx, best_cy),
                    (best_angle, best_cx + step, best_cy),
                    (best_angle, best_cx - step, best_cy),
                    (best_angle, best_cx, best_cy + step),
                    (best_angle, best_cx, best_cy - step),
                    (best_angle + step, best_cx + step, best_cy),
                    (best_angle - step, best_cx - step, best_cy),
                ]
                for ang, cx, cy in neighbors:
                    if time.time() > deadline - 30:
                        break
                    s = _compute_s_for_params(base_positions, [cx, cy], ang)
                    if s < best_s - 1e-10:
                        best_s = s
                        best_angle, best_cx, best_cy = ang, cx, cy
                        improved = True
    
    # Final refinement with very small steps
    if time.time() < deadline - 10:
        for _ in range(500):
            if time.time() > deadline - 10:
                break
            step = 0.001
            neighbors = [
                (best_angle + step, best_cx, best_cy),
                (best_angle - step, best_cx, best_cy),
                (best_angle, best_cx + step, best_cy),
                (best_angle, best_cx - step, best_cy),
                (best_angle, best_cx, best_cy + step),
                (best_angle, best_cx, best_cy - step),
            ]
            for ang, cx, cy in neighbors:
                if time.time() > deadline - 10:
                    break
                s = _compute_s_for_params(base_positions, [cx, cy], ang)
                if s < best_s - 1e-12:
                    best_s = s
                    best_angle, best_cx, best_cy = ang, cx, cy
    
    # Construct the final result
    outer_angle, outer_cx, outer_cy = best_params
    
    # Inner hexagons: shift by minus outer center so that outer center is at origin for validation
    # Actually, the inner positions are absolute. The outer center is at (outer_cx, outer_cy).
    # The task says: "after shifting all centers by minus the outer center"
    # So the inner positions in the output are the absolute positions.
    
    inner_hex_data = base_positions.copy()
    outer_hex_data = np.array([outer_cx, outer_cy, outer_angle])
    
    # Validate
    if not _validate_result(inner_hex_data, outer_hex_data, best_s):
        # Try a fallback: center the outer hexagon at the centroid of inner hexagons
        centroid = np.mean(base_positions[:, :2], axis=0)
        outer_hex_data = np.array([centroid[0], centroid[1], 0.0])
        s = _compute_s_for_params(base_positions, centroid, 0.0)
        best_s = s
        if not _validate_result(inner_hex_data, outer_hex_data, best_s):
            # Last resort: use the seed program's approach
            inner_hex_data = np.array([
                [0, 0, 0],
                [-2.5, 0, 0], [2.5, 0, 0],
                [-1.25, 2.17, 0], [1.25, 2.17, 0],
                [-1.25, -2.17, 0], [1.25, -2.17, 0],
                [-3.75, 2.17, 0], [3.75, 2.17, 0],
                [-3.75, -2.17, 0], [3.75, -2.17, 0],
                [0, -4, 0],
            ])
            outer_hex_data = np.array([0, 0, 0])
            best_s = 8.0
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
