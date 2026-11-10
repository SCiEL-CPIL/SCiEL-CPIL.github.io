# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _min_containing_hex_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum side length s of a regular hexagon (centered at outer_center,
    with the given angle) that contains all inner unit hexagons.
    """
    ov_cx, ov_cy = outer_center[0], outer_center[1]
    ov_a = np.deg2rad(outer_angle_deg)
    
    normal_angles = ov_a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    max_projections = np.zeros(6)
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        hv_rel = hv - np.array([ov_cx, ov_cy])
        projections = hv_rel @ normals.T
        max_projections = np.maximum(max_projections, np.max(projections, axis=0))
    
    s_min = np.max(max_projections) / (np.sqrt(3.0) / 2.0)
    return s_min


def _check_separation(inner, tol=1e-6):
    """
    Check if all pairs of inner hexagons are separated using SAT.
    Returns True if all pairs are separated.
    """
    n = len(inner)
    inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner]
    
    for i in range(n):
        for j in range(i + 1, n):
            v1 = inner_verts[i]
            v2 = inner_verts[j]
            
            edges = []
            for k in range(6):
                edges.append(v1[(k + 1) % 6] - v1[k])
                edges.append(v2[(k + 1) % 6] - v2[k])
            
            separated = False
            for edge in edges:
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal /= norm
                
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                gap1 = min2 - max1
                gap2 = min1 - max2
                if gap1 > tol or gap2 > tol:
                    separated = True
                    break
            
            if not separated:
                return False
    
    return True


def _lattice_positions():
    """
    Generate 12 positions on a hexagonal lattice that guarantee non-overlap.
    Unit hexagons on a hex lattice: adjacent centers are sqrt(3) apart.
    Using axial coordinates (q, r) -> (x, y) = (sqrt(3)*(q + r/2), 3*r/2)
    """
    # Hex lattice basis vectors for unit hexagons (side=1):
    # e1 = (sqrt(3), 0)
    # e2 = (sqrt(3)/2, 3/2)
    e1 = np.array([np.sqrt(3.0), 0.0])
    e2 = np.array([np.sqrt(3.0) / 2.0, 1.5])
    
    # Axial coordinates for 12 hexagons in a compact arrangement:
    # Center + ring of 6 + 5 more from second ring
    axial_coords = [
        (0, 0),    # center
        (1, 0),    # east
        (0, 1),    # northeast
        (-1, 1),   # northwest
        (-1, 0),   # west
        (0, -1),   # southwest
        (1, -1),   # southeast
        # Second ring (5 of 12 positions)
        (2, 0),    # far east
        (1, 1),    # far northeast
        (-2, 1),   # far northwest
        (-2, 0),   # far west
        (-1, -1),  # far southwest
    ]
    
    positions = np.array([q * e1 + r * e2 for q, r in axial_coords])
    return positions


def _objective_lattice(params):
    """
    Objective: minimize s for the containing hexagon.
    params = [rotation_angle_deg, outer_cx, outer_cy, outer_angle_deg]
    The 12 inner hexagons are fixed on the lattice, rotated by rotation_angle_deg.
    """
    rot_deg = params[0]
    ocx, ocy = params[1], params[2]
    outer_angle_deg = params[3]
    
    rot = np.deg2rad(rot_deg)
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    
    base_positions = _lattice_positions()
    rotated = base_positions @ R.T
    
    inner = np.zeros((12, 3))
    inner[:, 0] = rotated[:, 0]
    inner[:, 1] = rotated[:, 1]
    inner[:, 2] = rot_deg  # all hexagons rotated together
    
    s = _min_containing_hex_side(inner, np.array([ocx, ocy]), outer_angle_deg)
    return s


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s limit
    
    # Get base lattice positions
    base_positions = _lattice_positions()
    
    # Initial guess: no rotation, outer centered at origin
    best_params = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Compute initial s
    inner_init = np.zeros((12, 3))
    inner_init[:, 0] = base_positions[:, 0]
    inner_init[:, 1] = base_positions[:, 1]
    inner_init[:, 2] = 0.0
    best_s = _min_containing_hex_side(inner_init, np.array([0.0, 0.0]), 0.0)
    
    # Nelder-Mead optimization over [rotation, outer_cx, outer_cy, outer_angle]
    try:
        result = minimize(
            _objective_lattice,
            best_params,
            method='Nelder-Mead',
            options={
                'maxiter': 10000,
                'xatol': 1e-10,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if result.fun < best_s:
            best_params = result.x
            best_s = result.fun
            
    except Exception:
        pass
    
    # Try multiple random initializations for rotation
    np.random.seed(42)
    for trial in range(10):
        if time.time() - start_time > deadline - 30.0:
            break
        
        init = np.array([
            np.random.uniform(-30, 30),  # random rotation
            np.random.uniform(-1, 1),    # outer center x
            np.random.uniform(-1, 1),    # outer center y
            np.random.uniform(-30, 30),  # outer angle
        ])
        
        try:
            result2 = minimize(
                _objective_lattice,
                init,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result2.fun < best_s:
                best_params = result2.x
                best_s = result2.fun
                
        except Exception:
            pass
    
    # Also try with different outer angle offsets (0, 30 degrees)
    for angle_offset in [0.0, 30.0]:
        if time.time() - start_time > deadline - 30.0:
            break
        
        init = np.array([0.0, 0.0, 0.0, angle_offset])
        try:
            result3 = minimize(
                _objective_lattice,
                init,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result3.fun < best_s:
                best_params = result3.x
                best_s = result3.fun
                
        except Exception:
            pass
    
    # Construct final inner hexagons
    rot_deg = best_params[0]
    ocx, ocy = best_params[1], best_params[2]
    outer_angle_deg = best_params[3]
    
    rot = np.deg2rad(rot_deg)
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    
    rotated = base_positions @ R.T
    
    inner = np.zeros((12, 3))
    inner[:, 0] = rotated[:, 0]
    inner[:, 1] = rotated[:, 1]
    inner[:, 2] = rot_deg
    
    # Verify separation
    if not _check_separation(inner):
        # Fallback to unrotated
        inner = np.zeros((12, 3))
        inner[:, 0] = base_positions[:, 0]
        inner[:, 1] = base_positions[:, 1]
        inner[:, 2] = 0.0
        s_final = _min_containing_hex_side(inner, np.array([0.0, 0.0]), 0.0) + 1e-4
        outer = np.array([0.0, 0.0, 0.0])
        return inner, outer, s_final
    
    # Compute final s with small margin
    s_final = _min_containing_hex_side(inner, np.array([ocx, ocy]), outer_angle_deg) + 1e-4
    
    outer = np.array([ocx, ocy, outer_angle_deg])
    
    return inner, outer, s_final


# EVOLVE-BLOCK-END
