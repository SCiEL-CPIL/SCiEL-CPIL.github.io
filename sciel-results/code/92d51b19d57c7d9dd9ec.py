# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _compute_min_side_fast(inner_positions, outer_center, outer_angle_deg):
    """
    Fast computation of min side length s of outer hexagon containing inner hexagons.
    inner_positions: (12, 2) centers. Inner hexagons have side 1, angle 0.
    outer_center: (2,)
    outer_angle_deg: float
    """
    outer_angle = np.deg2rad(outer_angle_deg)
    # Normals to the edges of the outer hexagon
    # Vertices are at outer_angle + 60i. Edges are perpendicular to directions outer_angle + 30 + 60i
    normal_angles = outer_angle + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Vertices of unit hexagon (side 1, angle 0)
    unit_angles = np.deg2rad(60 * np.arange(6))
    unit_cos = np.cos(unit_angles)
    unit_sin = np.sin(unit_angles)
    unit_verts = np.column_stack([unit_cos, unit_sin]) # (6, 2)
    
    # All vertices of all 12 inner hexagons
    # (12, 6, 2)
    all_verts = inner_positions[:, np.newaxis, :] + unit_verts[np.newaxis, :, :]
    all_verts = all_verts.reshape(-1, 2) # (72, 2)
    
    # Relative to outer center
    rel_verts = all_verts - outer_center
    
    # Projections onto normals
    # (72, 6)
    projections = rel_verts @ normals.T
    
    # For each normal, we need the max and min projection
    max_projs = np.max(projections, axis=0)
    min_projs = np.min(projections, axis=0)
    
    # The inradius (apothem) R must be >= max(max_projs, -min_projs)
    # Actually, for a regular hexagon centered at origin, the boundary is |proj| <= R
    # So R >= max_k ( max(proj_k) , -min(proj_k) )
    # R = max_k max( max(proj_k), -min(proj_k) )
    R = np.max(np.maximum(max_projs, -min_projs))
    
    if R < 0:
        R = 0
    
    # Side length s = 2 * R / sqrt(3)
    s = 2.0 * R / np.sqrt(3.0)
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
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
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
    if not np.isfinite(s) or s < np.sqrt(12) - 1e-8:
        return False
    
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    # Check distinct vertices
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
                
    inner_verts_list = []
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        inner_verts_list.append(verts)
        # Check edge lengths
        for k in range(6):
            edge_len = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
                
    # Check separation
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
                
    # Check containment
    for i in range(12):
        for v in inner_verts_list[i]:
            if not _vertex_inside_outer(v, outer_verts):
                return False
    return True

def _generate_topology_12_hex_lattice():
    """
    Generates 12 positions in a hexagonal lattice.
    Basis: u = (sqrt(3), 0), v = (sqrt(3)/2, 1.5)
    """
    sqrt3 = np.sqrt(3)
    positions = []
    # Ring 0
    positions.append([0, 0])
    # Ring 1 (6 neighbors)
    # Directions: 0, 60, 120, 180, 240, 300
    # Using axial coordinates (q, r) -> x = sqrt3 * (q + r/2), y = 1.5 * r
    ring1_coords = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for q, r in ring1_coords:
        x = sqrt3 * (q + r/2.0)
        y = 1.5 * r
        positions.append([x, y])
        
    # We need 5 more. Let's add them from Ring 2.
    # Ring 2 coords (q, r) with max(abs(q), abs(r), abs(-q-r)) == 2
    # To make a compact shape, pick 5 from ring 2.
    # Common optimal shape for 12 is often a "loose" hexagon or a rectangle.
    # Let's try adding (2,0), (1,1), (0,2), (-1,2) and one more?
    # Let's just use a simple 3x4 grid for variety in the list of topologies.
    
    # Topology 1: Hex cluster (1 + 6 + 5 from ring 2)
    ring2_candidates = [(2,0), (2,-1), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-1,2), (0,2), (1,1)]
    # Pick 5 to complete the symmetry as much as possible?
    # Let's pick (2,0), (1,1), (0,2), (-1,2), (-2,1) -> forms an arc?
    # Or just take the first 5.
    for q, r in ring2_candidates[:5]:
        x = sqrt3 * (q + r/2.0)
        y = 1.5 * r
        positions.append([x, y])
        
    positions = np.array(positions)
    # Center it
    center = positions.mean(axis=0)
    positions -= center
    return positions

def _generate_topology_12_grid():
    """
    Generates 12 positions in a 3x4 grid (slanted).
    """
    sqrt3 = np.sqrt(3)
    positions = []
    for r in range(3):
        for c in range(4):
            x = sqrt3 * (c + r/2.0)
            y = 1.5 * r
            positions.append([x, y])
    positions = np.array(positions)
    center = positions.mean(axis=0)
    positions -= center
    return positions

def _generate_topology_12_rect():
    """
    Generates 12 positions in a 2x6 grid (slanted).
    """
    sqrt3 = np.sqrt(3)
    positions = []
    for r in range(2):
        for c in range(6):
            x = sqrt3 * (c + r/2.0)
            y = 1.5 * r
            positions.append([x, y])
    positions = np.array(positions)
    center = positions.mean(axis=0)
    positions -= center
    return positions

def _evaluate_rigid_cluster(local_pos, params, deadline):
    """
    params: [theta_rad, tx, ty]
    local_pos: (12, 2)
    """
    if time.time() > deadline:
        return 1e10
    theta = params[0]
    tx = params[1]
    ty = params[2]
    
    c = np.cos(theta)
    s = np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    
    # Rotate and translate
    inner_pos = (local_pos @ rot.T) + np.array([tx, ty])
    
    # Outer center is at (0,0) for calculation, angle 0?
    # Wait, we can optimize the outer angle too?
    # If we fix outer angle to 0, we just rotate the cluster.
    # Is it better to allow outer angle optimization?
    # Rotating the cluster by theta and keeping outer at 0 is equivalent to
    # keeping cluster at 0 and rotating outer by -theta.
    # Due to symmetry of the outer hexagon (60 deg), we only need theta in [0, 60).
    # So fixing outer angle to 0 and optimizing cluster rotation is sufficient.
    
    s_val = _compute_min_side_fast(inner_pos, np.array([0.0, 0.0]), 0.0)
    return s_val

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner_hex_data, outer_hex_data, s)
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    topologies = [
        _generate_topology_12_hex_lattice(),
        _generate_topology_12_grid(),
        _generate_topology_12_rect()
    ]
    
    best_s = np.inf
    best_params = None
    best_topology_idx = 0
    
    # Bounds for Differential Evolution
    # theta in [0, 2*pi], tx, ty in [-10, 10]
    bounds = [(0, 2*np.pi), (-10, 10), (-10, 10)]
    
    for t_idx, local_pos in enumerate(topologies):
        if time.time() > deadline - 30:
            break
            
        # Run Differential Evolution
        try:
            res = differential_evolution(
                _evaluate_rigid_cluster,
                bounds,
                args=(local_pos, deadline),
                maxiter=100,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False # We will polish manually
            )
            
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x
                best_topology_idx = t_idx
                
            # Nelder-Mead refinement
            if time.time() > deadline - 20:
                break
            res_nm = minimize(
                _evaluate_rigid_cluster,
                best_params,
                args=(topologies[best_topology_idx], deadline),
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000}
            )
            if res_nm.fun < best_s:
                best_s = res_nm.fun
                best_params = res_nm.x
                best_topology_idx = t_idx
                
        except Exception as e:
            pass

    if best_s >= np.inf:
        # Fallback to a known valid configuration
        local_pos = topologies[0]
        theta = 0
        tx = 0
        ty = 0
        inner_pos = local_pos
        best_s = _compute_min_side_fast(inner_pos, np.array([0.0, 0.0]), 0.0)
        best_params = np.array([0, 0, 0])
        best_topology_idx = 0

    # Construct final result
    local_pos = topologies[best_topology_idx]
    theta = best_params[0]
    tx = best_params[1]
    ty = best_params[2]
    
    c = np.cos(theta)
    s_rot = np.sin(theta)
    rot = np.array([[c, -s_rot], [s_rot, c]])
    
    inner_positions = (local_pos @ rot.T) + np.array([tx, ty])
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = inner_positions[i, 0]
        inner_hex_data[i, 1] = inner_positions[i, 1]
        inner_hex_data[i, 2] = 0.0
        
    # Outer hexagon: Center (0,0), Angle 0, Side best_s
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    # Validate
    if not _validate(inner_hex_data, outer_hex_data, best_s):
        # If invalid, try to fix by slightly increasing s or shifting
        # This should rarely happen if DE converged properly
        best_s += 0.1
        if not _validate(inner_hex_data, outer_hex_data, best_s):
             # Fallback to grid
             local_pos = topologies[1]
             inner_positions = local_pos
             inner_hex_data = np.zeros((12, 3))
             for i in range(12):
                 inner_hex_data[i, 0] = inner_positions[i, 0]
                 inner_hex_data[i, 1] = inner_positions[i, 1]
                 inner_hex_data[i, 2] = 0.0
             outer_hex_data = np.array([0.0, 0.0, 0.0])
             best_s = _compute_min_side_fast(inner_positions, np.array([0.0, 0.0]), 0.0) + 0.1

    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
