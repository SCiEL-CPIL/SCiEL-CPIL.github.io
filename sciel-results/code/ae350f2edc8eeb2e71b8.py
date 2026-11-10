# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _compute_min_side(inner_positions, outer_center, outer_angle_deg):
    """
    Given inner hexagon centers (12,2) and outer hexagon center and angle,
    compute the minimum side length s of a regular hexagon centered at outer_center
    with the given angle that contains all inner unit hexagons (with angle 0).
    """
    outer_angle = np.deg2rad(outer_angle_deg)
    
    # Outward normals of the outer hexagon edges
    # Vertices are at angles: outer_angle + 60*i
    # Edge i connects vertex i to vertex i+1
    # Outward normal to edge i is at angle: outer_angle + 30 + 60*i
    normal_angles = outer_angle + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6,2)
    
    # Compute all 72 vertices of the 12 inner unit hexagons (all with angle 0)
    # Unit hexagon vertices at angle 0
    unit_angles = np.deg2rad(60 * np.arange(6))
    unit_cos = np.cos(unit_angles)
    unit_sin = np.sin(unit_angles)
    
    all_verts = np.zeros((12, 6, 2))
    for i in range(12):
        all_verts[i, :, 0] = inner_positions[i, 0] + unit_cos
        all_verts[i, :, 1] = inner_positions[i, 1] + unit_sin
    
    all_verts = all_verts.reshape(72, 2)
    
    # For each outward normal, compute max projection of (v - outer_center)
    rel = all_verts - outer_center  # (72, 2)
    max_projs = np.max(rel @ normals.T, axis=0)  # (6,)
    
    # inradius = s * sqrt(3)/2, so s = 2 * max_inradius / sqrt(3)
    max_inradius = np.max(max_projs)
    if max_inradius < 0:
        max_inradius = 0
    s = 2.0 * max_inradius / np.sqrt(3.0)
    return s

def _check_separation(hex1_verts, hex2_verts, tol=1e-6):
    """Separating axis test: return True if hexagons are separated (no overlap)."""
    axes = []
    for verts in (hex1_verts, hex2_verts):
        for i in range(6):
            edge = verts[(i + 1) % 6] - verts[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                axes.append(n / norm)
    
    for axis in axes:
        proj1 = hex1_verts @ axis
        proj2 = hex2_verts @ axis
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    return False

def _vertex_inside_outer(v, outer_verts, tol=1e-6):
    """Check if vertex v is inside or on the boundary of outer hexagon."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
        if cross < -tol:
            return False
    return True

def _validate(inner_hex_data, outer_hex_data, s):
    """Validate the packing. Return True if valid."""
    if s < np.sqrt(12) - 1e-8:
        return False
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for k in range(6):
            edge_len = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    inner_verts_list = []
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        inner_verts_list.append(verts)
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
    for i in range(12):
        for v in inner_verts_list[i]:
            if not _vertex_inside_outer(v, outer_verts):
                return False
    return True

def _make_initial_grid(n_rows, n_cols, dx, dy):
    """Create a grid of hexagon positions."""
    positions = []
    for row in range(n_rows):
        for col in range(n_cols):
            x = (col - (n_cols - 1) / 2.0) * dx
            y = (row - (n_rows - 1) / 2.0) * dy
            if row % 2 == 1:
                x += dx / 2
            positions.append([x, y])
    positions = np.array(positions)
    center = positions.mean(axis=0)
    positions -= center
    return positions

def _optimize_from_init(init_positions, outer_center_init, outer_angle_deg, deadline, maxiter=3000):
    """Run Nelder-Mead optimization from given initial positions."""
    sqrt3 = np.sqrt(3)
    
    def objective(p):
        inner_pos = p[:24].reshape(12, 2)
        oc = p[24:26]
        return _compute_min_side(inner_pos, oc, outer_angle_deg)
    
    init_params = np.concatenate([init_positions.flatten(), outer_center_init])
    
    result = minimize(objective, init_params, method='Nelder-Mead',
                      options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11})
    
    if time.time() > deadline:
        return None
    
    return result

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner_hex_data, outer_hex_data, s)
    """
    start_time = time.time()
    deadline = start_time + 570.0  # 30s safety margin before 600s limit
    
    sqrt3 = np.sqrt(3)
    
    # Multiple initial configurations to try
    initial_configs = []
    
    # Config 1: 4x3 grid
    dx1 = 2.5
    dy1 = sqrt3
    pos1 = _make_initial_grid(3, 4, dx1, dy1)
    initial_configs.append(pos1)
    
    # Config 2: 3x4 grid (transposed)
    dx2 = 2.5
    dy2 = sqrt3
    pos2 = _make_initial_grid(4, 3, dx2, dy2)
    initial_configs.append(pos2)
    
    # Config 3: Tighter 4x3 grid
    dx3 = 2.3
    dy3 = sqrt3 * 0.95
    pos3 = _make_initial_grid(3, 4, dx3, dy3)
    initial_configs.append(pos3)
    
    # Config 4: Hexagonal cluster - 1 center + 6 around + 5 more
    # Center at origin, 6 at distance 2.5, then 5 more at distance ~4.5
    pos4 = np.zeros((12, 2))
    pos4[0] = [0, 0]
    for i in range(6):
        angle = np.deg2rad(60 * i)
        pos4[i+1, 0] = 2.5 * np.cos(angle)
        pos4[i+1, 1] = 2.5 * np.sin(angle)
    # Add 5 more in a ring
    for i in range(5):
        angle = np.deg2rad(30 + 72 * i)
        pos4[i+7, 0] = 4.5 * np.cos(angle)
        pos4[i+7, 1] = 4.5 * np.sin(angle)
    center4 = pos4.mean(axis=0)
    pos4 -= center4
    initial_configs.append(pos4)
    
    # Config 5: Two rows of 6
    pos5 = np.zeros((12, 2))
    for i in range(6):
        pos5[i, 0] = (i - 2.5) * 2.5
        pos5[i, 1] = -sqrt3/2
        pos5[i+6, 0] = (i - 2.5) * 2.5 + 1.25
        pos5[i+6, 1] = sqrt3/2
    center5 = pos5.mean(axis=0)
    pos5 -= center5
    initial_configs.append(pos5)
    
    # Try different outer angles
    outer_angles = [0, 30, 60, 90, 120, 150]
    
    best_s = np.inf
    best_params = None
    best_angle = 0
    
    for angle_deg in outer_angles:
        if time.time() > deadline:
            break
        
        for init_pos in initial_configs:
            if time.time() > deadline:
                break
            
            result = _optimize_from_init(init_pos, np.array([0.0, 0.0]), angle_deg, deadline, maxiter=2000)
            
            if result is not None and result.fun < best_s:
                best_s = result.fun
                best_params = result.x
                best_angle = angle_deg
    
    # If we still have time, try to refine the best solution
    if best_params is not None and time.time() < deadline - 60:
        def objective_refine(p):
            inner_pos = p[:24].reshape(12, 2)
            oc = p[24:26]
            return _compute_min_side(inner_pos, oc, best_angle)
        
        result = minimize(objective_refine, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.fun < best_s:
            best_s = result.fun
            best_params = result.x
    
    # Extract best solution
    inner_positions = best_params[:24].reshape(12, 2)
    outer_center = best_params[24:26]
    
    # Build return arrays
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = inner_positions[i, 0]
        inner_hex_data[i, 1] = inner_positions[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([outer_center[0], outer_center[1], float(best_angle)])
    
    s_final = best_s
    
    # Validate
    if not _validate(inner_hex_data, outer_hex_data, s_final):
        # Fallback to a known valid configuration
        inner_hex_data = np.zeros((12, 3))
        positions_fb = initial_configs[0].copy()
        for i in range(12):
            inner_hex_data[i, 0] = positions_fb[i, 0]
            inner_hex_data[i, 1] = positions_fb[i, 1]
            inner_hex_data[i, 2] = 0.0
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        s_final = _compute_min_side(positions_fb, np.array([0.0, 0.0]), 0.0) + 0.2
    
    return inner_hex_data, outer_hex_data, s_final


# EVOLVE-BLOCK-END
