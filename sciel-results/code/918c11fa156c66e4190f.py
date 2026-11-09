import numpy as np
import time

def _hex_vertices(cx, cy, d, a):
    a_rad = np.deg2rad(a)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_outer_side(positions, angle_deg):
    """
    Compute the minimum side length of a regular hexagon with the given angle
    that encloses all unit hexagons at the given positions.
    """
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    all_verts = positions[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    
    a_rad = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        n_angle = a_rad + i * np.pi/3 + np.pi/6
        n_unit = np.array([np.cos(n_angle), np.sin(n_angle)])
        normals.append(n_unit)
    normals = np.array(normals)
    
    projections = all_verts_flat @ normals.T
    max_projections = np.max(projections, axis=0)
    min_s = np.max(max_projections) * 2 / np.sqrt(3)
    return min_s

def _check_validity(inner, outer, s):
    """
    Validates the solution according to the task contract.
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
                
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False

    inner_verts_list = []
    for i in range(12):
        cx, cy, ang = inner[i]
        inner_verts_list.append(_hex_vertices(cx, cy, 1, ang))
        
    for i in range(12):
        for j in range(i+1, 12):
            va = inner_verts_list[i]
            vb = inner_verts_list[j]
            separated = False
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

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590
    
    best_score = 0.0
    best_inner = None
    best_outer = None
    best_s = None
    
    h = np.sqrt(3)/2
    
    # Define a few rigid honeycomb cluster configurations
    # Config 1: 3-4-3-2 arrangement (compact)
    # Row 0 (y=0): 3 hexes at x = -h, 0, h
    # Row 1 (y=1.5): 4 hexes at x = -1.5h, -0.5h, 0.5h, 1.5h
    # Row 2 (y=3): 3 hexes at x = -h, 0, h
    # Row 3 (y=4.5): 2 hexes at x = -0.5h, 0.5h
    config1 = np.array([
        [-h, 0], [0, 0], [h, 0],
        [-1.5*h, 1.5], [-0.5*h, 1.5], [0.5*h, 1.5], [1.5*h, 1.5],
        [-h, 3], [0, 3], [h, 3],
        [-0.5*h, 4.5], [0.5*h, 4.5]
    ])
    
    # Config 2: 1-6-5 arrangement (hexagonal cluster)
    config2 = np.array([
        [0, 0],
        [2*h, 0], [h, 1.5], [-h, 1.5], [-2*h, 0], [-h, -1.5], [h, -1.5],
        [3*h, 1.5], [0, 3], [-3*h, 1.5], [-3*h, -1.5], [3*h, -1.5]
    ])
    
    # Config 3: 4-4-4 grid (rectangular)
    config3 = np.array([
        [0, 0], [2*h, 0], [4*h, 0], [6*h, 0],
        [h, 1.5], [3*h, 1.5], [5*h, 1.5], [7*h, 1.5],
        [0, 3], [2*h, 3], [4*h, 3], [6*h, 3]
    ])
    
    # Config 4: 2-3-4-3 arrangement
    config4 = np.array([
        [-0.5*h, 0], [0.5*h, 0],
        [-1.5*h, 1.5], [-0.5*h, 1.5], [0.5*h, 1.5],
        [-2.5*h, 3], [-1.5*h, 3], [-0.5*h, 3], [0.5*h, 3],
        [-1.5*h, 4.5], [-0.5*h, 4.5], [0.5*h, 4.5]
    ])
    
    configs = [config1, config2, config3, config4]
    
    # For each config, search over global rotation and outer hexagon orientation
    for cfg_idx, base_pos in enumerate(configs):
        if time.time() > deadline:
            break
            
        # Center the cluster
        centroid = np.mean(base_pos, axis=0)
        centered_pos = base_pos - centroid
        
        # Search over rotation angles (0 to 60 degrees, since hexagons have 60-degree symmetry)
        num_angles = 200
        for a_idx in range(num_angles):
            if time.time() > deadline:
                    break
                    
            rot_angle = a_idx * 60.0 / num_angles
            rot_rad = np.deg2rad(rot_angle)
            rot_matrix = np.array([
                [np.cos(rot_rad), -np.sin(rot_rad)],
                [np.sin(rot_rad), np.cos(rot_rad)]
            ])
            
            rotated_pos = (centered_pos @ rot_matrix.T)
            
            # Search over outer hexagon orientation (0 to 60 degrees)
            num_outer_angles = 50
            for oa_idx in range(num_outer_angles):
                if time.time() > deadline:
                        break
                        
                outer_angle = oa_idx * 60.0 / num_outer_angles
                s = _min_outer_side(rotated_pos, outer_angle)
                
                if s < np.sqrt(12):
                    continue
                    
                score_candidate = 1.0 / s
                if score_candidate > best_score:
                    best_score = score_candidate
                    
                    # Construct inner array (all inner hexes have 0 angle relative to global frame)
                    inner_candidate = np.zeros((12, 3))
                    inner_candidate[:, :2] = rotated_pos
                    inner_candidate[:, 2] = 0.0
                    
                    outer_candidate = np.array([0, 0, outer_angle])
                    
                    # Final validity check
                    if _check_validity(inner_candidate, outer_candidate, s):
                        best_inner = inner_candidate
                        best_outer = outer_candidate
                        best_s = s
    
    if best_inner is not None:
        return best_inner, best_outer, best_s
    else:
        # Fallback: use config1 with angle 0
        base_pos = configs[0]
        centroid = np.mean(base_pos, axis=0)
        centered_pos = base_pos - centroid
        s = _min_outer_side(centered_pos, 0)
        if s < np.sqrt(12):
            s = np.sqrt(12)
        inner_fallback = np.zeros((12, 3))
        inner_fallback[:, :2] = centered_pos
        inner_fallback[:, 2] = 0.0
        outer_fallback = np.array([0, 0, 0])
        return inner_fallback, outer_fallback, s
