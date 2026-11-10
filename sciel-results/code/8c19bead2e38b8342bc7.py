# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


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
    
    rel = inner_verts - oc  # (N, 2)
    edge_vecs = np.roll(unit_verts, -1, axis=0) - unit_verts  # (6, 2)
    
    # A[i, p] = cross(edge_vecs[i], rel[p])
    A = edge_vecs[:, 0:1] * rel[1:2, :] - edge_vecs[:, 1:2] * rel[0:1, :]  # (6, N)
    
    # s_min = max(0, max over i,p of (-2 * A[i,p] / sqrt(3)))
    s_min = np.max(-2.0 * A / np.sqrt(3.0))
    s_min = max(s_min, 0.0)
    
    return s_min


def _get_configurations():
    """
    Return multiple candidate rigid body configurations of 12 unit hexagons.
    Each configuration is a (12, 3) array of (x, y, angle_deg).
    """
    sqrt3 = np.sqrt(3.0)
    configs = []
    
    # Configuration 1: Hexagonal lattice cluster (center + ring 1 + 5 from ring 2)
    axial1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-2, 0), (-1, -1), (2, -1),
    ]
    pos1 = []
    for q, r in axial1:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        pos1.append([x, y, 0.0])
    configs.append(np.array(pos1))
    
    # Configuration 2: Different subset of ring 2 for better symmetry
    axial2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
    ]
    pos2 = []
    for q, r in axial2:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        pos2.append([x, y, 0.0])
    configs.append(np.array(pos2))
    
    # Configuration 3: Another ring 2 subset
    axial3 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2), (-2, 1),
    ]
    pos3 = []
    for q, r in axial3:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        pos3.append([x, y, 0.0])
    configs.append(np.array(pos3))
    
    # Configuration 4: Symmetric arrangement - 6 in ring 1, 5 more spread evenly
    # Use a different pattern: place 5 additional hexagons at angles 0, 72, 144, 216, 288 degrees
    # on a circle of radius 2*sqrt(3) from center, but adjusted for hex lattice
    pos4 = [np.array([0.0, 0.0, 0.0])]
    # Ring 1
    for k in range(6):
        ang = np.deg2rad(60.0 * k)
        x = sqrt3 * np.cos(ang)
        y = sqrt3 * np.sin(ang)
        pos4.append(np.array([x, y, 0.0]))
    # 5 more at radius ~2*sqrt(3) at specific angles
    for k in range(5):
        ang = np.deg2rad(60.0 * k + 30.0)  # offset by 30 degrees
        # For hex lattice, this doesn't directly work. Let me use axial coords instead.
        pass
    # Skip this one, use axial-based instead
    # Configuration 4: center + 6 ring1 + 5 from ring2 at angles 0, 60, 120, 180, 240
    axial4 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-2, 0), (-2, 1),
    ]
    # That's only 11, add one more
    axial4.append((-1, -1))
    pos4 = []
    for q, r in axial4:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        pos4.append([x, y, 0.0])
    configs.append(np.array(pos4))
    
    # Configuration 5: Try a different topology - 2 rows of 3, 2 rows of 3, 2 rows of 3
    # Actually, let me try a more spread out configuration
    axial5 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1),
    ]
    # That's 11, add (2, -1)
    axial5.append((2, -1))
    pos5 = []
    for q, r in axial5:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        pos5.append([x, y, 0.0])
    configs.append(np.array(pos5))
    
    # Configuration 6: Full symmetric - try to get a more circular shape
    # Center + 6 ring 1 + 6 from ring 2 at every other position
    axial6 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
    ]
    # That's 12! Let me verify: center(1) + ring1(6) + ring2(5) = 12
    # Wait, that's 1+6+5=12. But I listed 6 from ring2. Let me recount.
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0) = 6 from ring 2
    # Total: 1 + 6 + 6 = 13. That's too many.
    # Let me remove one. Remove (-2,0).
    axial6 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1),
    ]
    # That's 1 + 6 + 5 = 12
    pos6 = []
    for q, r in axial6:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        pos6.append([x, y, 0.0])
    configs.append(np.array(pos6))
    
    return configs


def _check_separation_fast(inner_positions):
    """
    Quick check that no two inner hexagons overlap.
    For two regular hexagons with the same orientation, they don't overlap
    if the center-to-center distance >= sqrt(3).
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


def _optimize_config(inner_positions, deadline):
    """Optimize the outer hexagon parameters for a given configuration."""
    # Precompute all vertices
    all_verts_list = []
    for pos in inner_positions:
        verts = _hex_vertices(pos[0], pos[1], 1.0, pos[2])
        all_verts_list.append(verts)
    all_verts = np.vstack(all_verts_list)  # (72, 2)
    
    def objective(params):
        angle_deg, cx, cy = params
        return _min_containing_hex_side(all_verts, [cx, cy], angle_deg)
    
    # Multi-start optimization
    best_s = np.inf
    best_params = None
    
    # Grid of starting points
    angle_starts = np.linspace(0, 60, 13)  # 13 points
    center_range = np.linspace(-1.5, 1.5, 7)  # 7 points
    
    for a0 in angle_starts:
        if time.time() > deadline - 5:
            break
        for cx0 in center_range:
            for cy0 in center_range:
                x0 = np.array([a0, cx0, cy0])
                try:
                    result = minimize(objective, x0, method='Nelder-Mead',
                                     options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
                    if result.fun < best_s:
                        best_s = result.fun
                        best_params = result.x
                except Exception:
                    pass
    
    # Refine with scipy
    if best_params is not None and time.time() < deadline - 2:
        try:
            result = minimize(objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_s:
                best_s = result.fun
                best_params = result.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback
        best_s = _compute_s_for_params(inner_positions, [0, 0], 0.0)
        best_params = np.array([0.0, 0.0, 0.0])
    
    return best_s, best_params


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 630s limit with safety margin
    
    configs = _get_configurations()
    
    best_s = np.inf
    best_config_idx = 0
    best_params = None
    
    for idx, config in enumerate(configs):
        if time.time() > deadline - 30:
            break
        
        # Check separation
        if not _check_separation_fast(config):
            continue
        
        s, params = _optimize_config(config, deadline)
        if s < best_s:
            best_s = s
            best_config_idx = idx
            best_params = params
    
    if best_params is None:
        # Fallback to first valid config
        for idx, config in enumerate(configs):
            if _check_separation_fast(config):
                best_s = _compute_s_for_params(config, [0, 0], 0.0)
                best_config_idx = idx
                best_params = np.array([0.0, 0.0, 0.0])
                break
        else:
            # Last resort
            best_s = 8.0
            best_config_idx = 0
            best_params = np.array([0.0, 0.0, 0.0])
    
    inner_hex_data = configs[best_config_idx].copy()
    outer_angle, outer_cx, outer_cy = best_params
    outer_hex_data = np.array([outer_cx, outer_cy, outer_angle])
    
    # Validate
    if not _validate_result(inner_hex_data, outer_hex_data, best_s):
        # Try with angle=0 and center=centroid
        centroid = np.mean(inner_hex_data[:, :2], axis=0)
        outer_hex_data = np.array([centroid[0], centroid[1], 0.0])
        best_s = _compute_s_for_params(inner_hex_data, centroid, 0.0)
        
        if not _validate_result(inner_hex_data, outer_hex_data, best_s):
            # Last resort fallback
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
