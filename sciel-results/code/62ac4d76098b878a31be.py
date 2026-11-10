# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])


def _outer_side_length_from_positions(positions, angles_deg):
    """
    Given inner hexagon centers (positions, shape (12,2)) and angles (shape (12,)),
    compute the minimum outer hexagon side length (centered at origin, angle 0)
    that contains all inner hexagons.
    """
    n = len(positions)
    # Outer hexagon with side s, centered at origin, angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # For containment, each inner vertex must satisfy: for each outer edge,
    # cross(p2-p1, v-p1) >= 0
    
    # The 6 outer edges of a regular hexagon with side s, centered at origin, angle 0:
    # Vertex i: (s*cos(60*i deg), s*sin(60*i deg))
    # Edge i goes from vertex i to vertex (i+1)%6
    
    # For a regular hexagon centered at origin with "pointy top" or "flat top"?
    # With angle 0, vertex 0 is at (s, 0), so flat-top orientation.
    # The 6 outward normals point at angles 30, 90, 150, 210, 270, 330 degrees
    # (perpendicular to each edge, pointing outward)
    
    # Actually, let's think about this differently.
    # A regular hexagon with side length s, centered at origin, with one vertex at angle 0:
    # The distance from center to each edge (apothem) is s * sqrt(3)/2
    # The 6 edge normals point at angles 30, 90, 150, 210, 270, 330 degrees
    
    # For containment: for each inner vertex v, and for each of the 6 edge normals n_i,
    # we need dot(v, n_i) <= apothem = s * sqrt(3)/2
    # So s >= dot(v, n_i) / (sqrt(3)/2) for all v, i
    # Thus s = max over all v, i of dot(v, n_i) / (sqrt(3)/2)
    
    # Edge normals (unit vectors) pointing outward:
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # shape (6,2)
    
    apothem_factor = np.sqrt(3) / 2  # apothem = s * apothem_factor
    
    max_proj = 0
    for i in range(n):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
        # verts shape (6, 2)
        # Project each vertex onto each normal
        projs = verts @ normals.T  # shape (6, 6)
        max_proj = max(max_proj, np.max(projs))
    
    s = max_proj / apothem_factor
    return s


def _check_validity(inner_hex_data, outer_hex_data, s, tol=1e-6):
    """Check if the packing is valid."""
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    # Check outer vertices are distinct
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    # Check inner hexagons are unit side
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for k in range(6):
            edge_len = np.linalg.norm(verts[(k+1)%6] - verts[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    # Check separation between inner hexagons (SAT)
    for i in range(12):
        for j in range(i+1, 12):
            verts_i = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
            verts_j = _hex_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1.0, inner_hex_data[j, 2])
            
            # Get edge normals for hex i
            for k in range(6):
                edge = verts_i[(k+1)%6] - verts_i[k]
                normal = np.array([-edge[1], edge[0]])
                normal = normal / np.linalg.norm(normal)
                
                proj_i = verts_i @ normal
                proj_j = verts_j @ normal
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                # If intervals are more than tol apart, separated on this axis
                if max_i + tol < min_j or max_j + tol < min_i:
                    break  # Found a separating axis, these two don't overlap
            else:
                # No separating axis found -> overlap
                return False
    
    # Check all inner vertices are inside outer hexagon
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for v in verts:
            # For each outer edge (p1, p2), check cross(p2-p1, v-p1) >= -tol
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < -tol:
                    inside = False
                    break
            if not inside:
                return False
    
    return True


def _objective(params):
    """Objective function: outer hexagon side length given parameters."""
    # params: 36 values = 12*(x, y, angle_deg)
    positions = params[0::3]
    ys = params[1::3]
    angles = params[2::3]
    
    pos_arr = np.column_stack([positions, ys])
    return _outer_side_length_from_positions(pos_arr, angles)


def _gradient(params):
    """Numerical gradient of the objective."""
    eps = 1e-7
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps
        grad[i] = (_objective(params_plus) - _objective(params_minus)) / (2*eps)
    return grad


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = 600.0  # seconds
    
    # Initial arrangement from seed
    initial_positions = np.array([
        [0, 0],
        [-2.5, 0],
        [2.5, 0],
        [-1.25, 2.17],
        [1.25, 2.17],
        [-1.25, -2.17],
        [1.25, -2.17],
        [-3.75, 2.17],
        [3.75, 2.17],
        [-3.75, -2.17],
        [3.75, -2.17],
        [0, -4]
    ])
    initial_angles = np.zeros(12)
    
    # Pack into params vector
    params = np.zeros(36)
    for i in range(12):
        params[3*i] = initial_positions[i, 0]
        params[3*i+1] = initial_positions[i, 1]
        params[3*i+2] = initial_angles[i]
    
    best_params = params.copy()
    best_s = _outer_side_length_from_positions(initial_positions, initial_angles)
    
    # Try a few different initial arrangements
    # Arrangement 1: Hexagonal close packing
    # Center at origin, 6 around it, then 6 more
    # For unit hexagons in hexagonal packing:
    # Horizontal spacing: 2 * (sqrt(3)/2) * 2 = 2*sqrt(3) ≈ 3.464 (for adjacent columns)
    # Actually for flat-top hexagons with side 1:
    # Width = 2*1 = 2, height = sqrt(3)*1 ≈ 1.732
    # Adjacent in same row: dx = 3, dy = 0 (since width is 2, gap is 1... wait)
    # Let me reconsider. Two unit hexagons side by side (flat top):
    # Each has width 2 (from -1 to 1 in x). If they share an edge, centers are 3 apart horizontally.
    # Vertical offset for adjacent row: dy = sqrt(3) ≈ 1.732, dx = 1.5
    
    # Standard hexagonal packing of unit hexagons:
    # Row 0: x = 0, 3, 6, ... ; y = 0
    # Row 1: x = 1.5, 4.5, ... ; y = sqrt(3)
    # Row 2: x = 0, 3, 6, ... ; y = 2*sqrt(3)
    
    # Let's try a 2-3-3-2-2 arrangement or similar
    # Actually, let's just optimize from multiple starting points
    
    starting_configs = []
    
    # Config 0: seed
    starting_configs.append((initial_positions, initial_angles))
    
    # Config 1: Tighter hexagonal packing
    # 12 hexagons in a hexagonal cluster
    # A hexagonal number arrangement: center + ring of 6 + partial ring of 6
    # But 12 = 1 + 6 + 5, so not a complete hexagon
    # Let's do: center, 6 around, 5 more around the outside
    sqrt3 = np.sqrt(3)
    hex_pos = np.array([
        [0, 0],
        [3, 0],
        [1.5, sqrt3],
        [-1.5, sqrt3],
        [-3, 0],
        [-1.5, -sqrt3],
        [1.5, -sqrt3],
        [4.5, sqrt3],
        [3, 2*sqrt3],
        [0, 2*sqrt3],
        [-3, 2*sqrt3],
        [-4.5, sqrt3]
    ])
    hex_angles = np.zeros(12)
    starting_configs.append((hex_pos, hex_angles))
    
    # Config 2: Rotated hexagons for better packing
    rot_pos = hex_pos.copy()
    rot_angles = np.zeros(12)
    rot_angles[1] = 30  # Rotate some
    rot_angles[2] = 30
    rot_angles[3] = 30
    rot_angles[4] = 30
    rot_angles[5] = 30
    rot_angles[6] = 30
    starting_configs.append((rot_pos, rot_angles))
    
    best_overall_s = float('inf')
    best_overall_params = None
    
    for config_idx, (pos, ang) in enumerate(starting_configs):
        if time.time() > deadline - 30:
            break
        
        p = np.zeros(36)
        for i in range(12):
            p[3*i] = pos[i, 0]
            p[3*i+1] = pos[i, 1]
            p[3*i+2] = ang[i]
        
        s_val = _outer_side_length_from_positions(pos, ang)
        
        # Simple gradient descent with adaptive step size
        step = 0.1
        for iteration in range(500):
            if time.time() > deadline - 30:
                break
            
            s_current = _outer_side_length_from_positions(
                np.column_stack([p[0::3], p[1::3]]),
                p[2::3]
            )
            
            if s_current < best_overall_s:
                best_overall_s = s_current
                best_overall_params = p.copy()
            
            # Compute gradient
            grad = np.zeros(36)
            eps = 1e-6
            for i in range(36):
                p_plus = p.copy()
                p_plus[i] += eps
                s_plus = _outer_side_length_from_positions(
                    np.column_stack([p_plus[0::3], p_plus[1::3]]),
                    p_plus[2::3]
                )
                grad[i] = (s_plus - s_current) / eps
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            
            # Normalize gradient
            grad = grad / grad_norm
            
            # Take step
            p_new = p - step * grad
            
            s_new = _outer_side_length_from_positions(
                np.column_stack([p_new[0::3], p_new[1::3]]),
                p_new[2::3]
            )
            
            if s_new < s_current:
                p = p_new
                # Increase step slightly
                step = min(step * 1.05, 0.5)
            else:
                # Decrease step
                step *= 0.7
                if step < 1e-6:
                    break
        
        # After optimization, verify and refine
        pos_final = np.column_stack([p[0::3], p[1::3]])
        ang_final = p[2::3]
        s_final = _outer_side_length_from_positions(pos_final, ang_final)
        
        if s_final < best_overall_s:
            best_overall_s = s_final
            best_overall_params = p.copy()
    
    # Use the best result found
    if best_overall_params is not None:
        p = best_overall_params
    else:
        p = params
    
    # Final: construct output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = p[3*i]
        inner_hex_data[i, 1] = p[3*i+1]
        inner_hex_data[i, 2] = p[3*i+2]
    
    # Center the outer hexagon at origin
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    s = _outer_side_length_from_positions(
        np.column_stack([p[0::3], p[1::3]]),
        p[2::3]
    )
    
    # Verify validity; if invalid, fall back to seed
    if not _check_validity(inner_hex_data, outer_hex_data, s):
        # Fall back to seed arrangement
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
                [0, -4, 0]
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        s = 8.0
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
