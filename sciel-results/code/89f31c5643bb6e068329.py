# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 620.0

def _elapsed():
    return time.time() - _START_TIME

def _hex_vertices(cx, cy, s, angle_deg):
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(np.arange(0, 360, 60))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _separating_axis_check(v1, v2):
    """Check if two convex polygons are separated using SAT."""
    # Get all edge normals from both polygons
    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        norms = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = norms / (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-15)
        return norms
    
    for verts in [v1, v2]:
        normals = get_normals(verts)
        for n in normals:
            proj1 = v1 @ n
            proj2 = v2 @ n
            if max(proj1) - min(proj2) > 1e-6 or max(proj2) - min(proj1) > 1e-6:
                return True
    return False

def _point_in_hexagon(p, vertices):
    for i in range(6):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
        if cross < -1e-6:
            return False
    return True

def _check_all_valid(inner_data, outer_data, side):
    if not (np.all(np.isfinite(inner_data)) and np.all(np.isfinite(outer_data)) and np.isfinite(side)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    
    inner_verts = []
    for i in range(11):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        inner_verts.append(v)
    
    ox, oy, oa = outer_data
    outer_verts = _hex_vertices(ox, oy, side, oa)
    
    # Pairwise separation
    for i in range(11):
        for j in range(i + 1, 11):
            if not _separating_axis_check(inner_verts[i], inner_verts[j]):
                return False
    
    # All inner vertices inside outer
    for i in range(11):
        for p in inner_verts[i]:
            if not _point_in_hexagon(p, outer_verts):
                return False
    
    return True

def _min_outer_hex_for_cluster(inner_pos, inner_angles, outer_angle):
    """Find minimal side and center for outer hexagon containing all inner hexagons."""
    all_verts = []
    for i in range(11):
        x, y = inner_pos[i]
        a = inner_angles[i]
        v = _hex_vertices(x, y, 1.0, a)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    outer_a_rad = np.deg2rad(outer_angle)
    normal_angles = outer_a_rad + np.deg2rad(30) + np.deg2rad(60) * np.arange(6)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    projections = all_verts @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    
    sqrt3 = np.sqrt(3)
    required_s = (max_proj - min_proj) / sqrt3
    s = np.max(required_s)
    
    rhs = (max_proj + min_proj) / 2
    c, _, _, _ = np.linalg.lstsq(normals, rhs, rcond=None)
    
    return s, c

def hexagon_packing_11():
    best_side = float('inf')
    best_inner = None
    best_outer = None
    
    sqrt3 = np.sqrt(3)
    
    def make_cluster(pattern, rotation):
        positions = [[0.0, 0.0]]
        angles = [0.0]
        
        for k in range(6):
            theta = np.deg2rad(60 * k)
            x = sqrt3 * np.cos(theta)
            y = sqrt3 * np.sin(theta)
            positions.append([x, y])
            angles.append(0.0)
        
        if pattern == 0:
            for k in range(4):
                theta = np.deg2rad(90 * k + 30)
                x = 2 * sqrt3 * np.cos(theta)
                y = 2 * sqrt3 * np.sin(theta)
                positions.append([x, y])
                angles.append(0.0)
        elif pattern == 1:
            for idx in [0, 1, 3, 4]:
                theta = np.deg2rad(60 * idx)
                x = 2 * sqrt3 * np.cos(theta)
                y = 2 * sqrt3 * np.sin(theta)
                positions.append([x, y])
                angles.append(0.0)
        elif pattern == 2:
            for k in range(4):
                theta = np.deg2rad(90 * k)
                x = 2 * sqrt3 * np.cos(theta)
                y = 2 * sqrt3 * np.sin(theta)
                positions.append([x, y])
                angles.append(0.0)
        elif pattern == 3:
            for k in range(4):
                theta = np.deg2rad(60 * k)
                x = 2 * sqrt3 * np.cos(theta)
                y = 2 * sqrt3 * np.sin(theta)
                positions.append([x, y])
                angles.append(0.0)
        elif pattern == 4:
            for k in range(3):
                theta = np.deg2rad(60 * k)
                x = 2 * sqrt3 * np.cos(theta)
                y = 2 * sqrt3 * np.sin(theta)
                positions.append([x, y])
                angles.append(0.0)
            theta = np.deg2rad(180)
            x = 2 * sqrt3 * np.cos(theta)
            y = 2 * sqrt3 * np.sin(theta)
            positions.append([x, y])
            angles.append(0.0)
        else:
            for k in range(2):
                theta = np.deg2rad(60 * k)
                x = 2 * sqrt3 * np.cos(theta)
                y = 2 * sqrt3 * np.sin(theta)
                positions.append([x, y])
                angles.append(0.0)
            for k in range(2):
                theta = np.deg2rad(120 + 60 * k)
                x = 2 * sqrt3 * np.cos(theta)
                y = 2 * sqrt3 * np.sin(theta)
                positions.append([x, y])
                angles.append(0.0)
        
        pos_arr = np.array(positions, dtype=float)
        ang_arr = np.array(angles, dtype=float)
        
        rot_rad = np.deg2rad(rotation)
        R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                      [np.sin(rot_rad), np.cos(rot_rad)]])
        pos_arr = pos_arr @ R.T
        
        return pos_arr, ang_arr
    
    # Search over patterns and rotations
    for pattern in range(6):
        for base_rot in np.linspace(0, 60, 13):
            if _elapsed() > _TIME_LIMIT:
                break
            
            pos_arr, ang_arr = make_cluster(pattern, base_rot)
            inner_data = np.column_stack([pos_arr, ang_arr])
            
            # Check if inner hexagons are non-overlapping
            inner_verts = []
            for i in range(11):
                x, y, a = inner_data[i]
                v = _hex_vertices(x, y, 1.0, a)
                inner_verts.append(v)
            
            overlap = False
            for i in range(11):
                for j in range(i + 1, 11):
                    if not _separating_axis_check(inner_verts[i], inner_verts[j]):
                        overlap = True
                        break
                if overlap:
                    break
            
            if overlap:
                continue
            
            # Try different outer hexagon angles
            for outer_angle in np.linspace(0, 60, 25):
                if _elapsed() > _TIME_LIMIT:
                    break
                
                s, center = _min_outer_hex_for_cluster(pos_arr, ang_arr, outer_angle)
                
                if s < best_side:
                    outer_data = np.array([center[0], center[1], outer_angle])
                    if _check_all_valid(inner_data, outer_data, s):
                        best_side = s
                        best_inner = inner_data.copy()
                        best_outer = outer_data.copy()
        
        if _elapsed() > _TIME_LIMIT:
            break
    
    if _elapsed() > _TIME_LIMIT:
        if best_inner is not None:
            return best_inner, best_outer, best_side
        # Fallback
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
            ]
        )
        return inner_hex_data, np.array([0, 0, 0]), 8.0
    
    if best_inner is not None:
        return best_inner, best_outer, best_side
    
    # Fallback
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
        ]
    )
    return inner_hex_data, np.array([0, 0, 0]), 8.0


# EVOLVE-BLOCK-END
