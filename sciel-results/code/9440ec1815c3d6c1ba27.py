import numpy as np
import time

N = 11
_DEADLINE = 140.0

def _hex_vertices(cx, cy, theta, side=1.0):
    angles = theta + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _min_distance_between_hexes(pos1, ang1, pos2, ang2, side=1.0):
    """
    Compute minimum distance between two regular hexagons.
    Returns the minimum distance (0 if overlapping).
    """
    v1 = _hex_vertices(pos1[0], pos1[1], ang1, side)
    v2 = _hex_vertices(pos2[0], pos2[1], ang2, side)
    
    # Check all vertex-vertex, vertex-edge distances
    min_d = float('inf')
    
    # Vertex to vertex
    for p in v1:
        for q in v2:
            d = np.linalg.norm(p - q)
            if d < min_d:
                min_d = d
    
    # Vertex to edge for hex1 vertices to hex2 edges
    for p in v1:
        for i in range(6):
            a = v2[i]
            b = v2[(i+1) % 6]
            ab = b - a
            ap = p - a
            t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
            t = max(0.0, min(1.0, t))
            closest = a + t * ab
            d = np.linalg.norm(p - closest)
            if d < min_d:
                min_d = d
    
    # Vertex to edge for hex2 vertices to hex1 edges
    for p in v2:
        for i in range(6):
            a = v1[i]
            b = v1[(i+1) % 6]
            ab = b - a
            ap = p - a
            t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
            t = max(0.0, min(1.0, t))
            closest = a + t * ab
            d = np.linalg.norm(p - closest)
            if d < min_d:
                min_d = d
    
    return min_d

def _packing_penalty(positions, angles, side=1.0):
    """
    Compute penalty for overlaps using exact hexagon distance.
    """
    pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _min_distance_between_hexes(positions[i], angles[i], positions[j], angles[j], side)
            if d < 1e-6:
                # Overlapping - heavy penalty
                pen += 100.0
            elif d < 0.01:
                # Very close - moderate penalty
                pen += 10.0 * (0.01 - d)
    return pen

def _outer_side(positions, angles, outer_center, outer_angle, side=1.0):
    """
    Returns side length of the minimum enclosing regular hexagon.
    """
    all_verts = []
    for i in range(N):
        v = _hex_vertices(positions[i, 0], positions[i, 1], angles[i], side)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    normals_angles = outer_angle + np.pi/6 + np.pi/3 * np.arange(6)
    normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
    
    proj = all_verts @ normals.T
    c_proj = outer_center @ normals.T
    diffs = np.abs(proj - c_proj)
    max_diff = diffs.max()
    outer_side = max_diff * 2.0 / np.sqrt(3.0)
    
    return outer_side

def _objective(v, outer_center, outer_angle):
    """
    v: [x1, y1, a1, ..., xN, yN, aN]
    """
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        positions[i, 0] = v[i*3]
        positions[i, 1] = v[i*3+1]
        angles[i] = v[i*3+2]
    
    pen = _packing_penalty(positions, angles)
    side = _outer_side(positions, angles, outer_center, outer_angle)
    
    return side + 50.0 * pen

def _simulate_annealing(v0, outer_center, outer_angle, start_temp=2.0, end_temp=0.01, n_iter=5000):
    """
    Simulated annealing to optimize hexagon positions and angles.
    """
    current_v = v0.copy()
    current_obj = _objective(current_v, outer_center, outer_angle)
    best_v = current_v.copy()
    best_obj = current_obj
    
    temp = start_temp
    cooling = (start_temp - end_temp) / n_iter
    
    for it in range(n_iter):
        # Generate perturbation
        new_v = current_v.copy()
        # Perturb one or two hexagons
        num_perturb = 1 if np.random.random() < 0.7 else 2
        for _ in range(num_perturb):
            idx = np.random.randint(0, N)
            # Perturb position
            new_v[idx*3] += np.random.randn() * 0.3
            new_v[idx*3+1] += np.random.randn() * 0.3
            # Perturb angle
            new_v[idx*3+2] += np.random.randn() * 0.2
        
        new_obj = _objective(new_v, outer_center, outer_angle)
        
        # Accept or reject
        if new_obj < current_obj:
            current_v = new_v
            current_obj = new_obj
            if new_obj < best_obj:
                best_v = new_v.copy()
                best_obj = new_obj
        else:
            delta = new_obj - current_obj
            if np.random.random() < np.exp(-delta / max(temp, 1e-8)):
                current_v = new_v
                current_obj = new_obj
        
        temp = max(end_temp, start_temp - cooling * it)
        
        # Check time
        if time.time() > _DEADLINE - 15:
            break
    
    return best_v, best_obj

def _generate_lattice_configs():
    """Generate hexagonal lattice configurations."""
    configs = []
    
    # Hex grid in axial coordinates
    axial_coords = [(0,0)]
    # Ring 1
    axial_coords += [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    # Ring 2 (first 4 to make 11 total)
    axial_coords += [(2,0), (1,1), (0,2), (-1,2)]
    
    selected_axial = axial_coords[:N]
    
    # Convert to Cartesian: x = 1.5*q, y = sqrt(3)*(r + q/2)
    lattice_centers = []
    for q, r in selected_axial:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2)
        lattice_centers.append((x, y))
    
    positions = np.array(lattice_centers)
    angles = np.zeros(N)
    cx = np.mean(positions[:, 0])
    cy = np.mean(positions[:, 1])
    theta = 0.0
    
    v = np.array([[x, y, a] for x, y, a in zip(positions[:,0], positions[:,1], angles)]).flatten()
    configs.append((v, np.array([cx, cy]), theta))
    
    # Alternative: slightly different arrangement
    axial_coords2 = [(0,0)]
    axial_coords2 += [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    axial_coords2 += [(2,0), (1,1), (0,2), (-2,1)]
    
    selected_axial2 = axial_coords2[:N]
    lattice_centers2 = []
    for q, r in selected_axial2:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2)
        lattice_centers2.append((x, y))
    
    positions2 = np.array(lattice_centers2)
    angles2 = np.zeros(N)
    cx2 = np.mean(positions2[:, 0])
    cy2 = np.mean(positions2[:, 1])
    theta2 = 0.0
    
    v2 = np.array([[x, y, a] for x, y, a in zip(positions2[:,0], positions2[:,1], angles2)]).flatten()
    configs.append((v2, np.array([cx2, cy2]), theta2))
    
    # Add random perturbations of the first config
    for _ in range(3):
        v_perturbed = v + np.random.randn(len(v)) * 0.2
        configs.append((v_perturbed, np.array([cx, cy]), theta))
    
    return configs

def construct():
    start_time = time.time()
    best_result = None
    best_side = float('inf')
    
    configs = _generate_lattice_configs()
    
    for v0, outer_center, outer_angle in configs:
        if time.time() > _DEADLINE - 20:
            break
        
        try:
            v_opt, obj_val = _simulate_annealing(v0, outer_center, outer_angle,
                                                  start_temp=1.5, end_temp=0.01, n_iter=4000)
            
            # Extract results
            positions = np.zeros((N, 2))
            angles = np.zeros(N)
            for i in range(N):
                positions[i, 0] = v_opt[i*3]
                positions[i, 1] = v_opt[i*3+1]
                angles[i] = v_opt[i*3+2]
            
            side = _outer_side(positions, angles, outer_center, outer_angle)
            
            # Validate: check no overlaps
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    d = _min_distance_between_hexes(positions[i], angles[i], positions[j], angles[j])
                    if d < 1e-4:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and side < best_side:
                best_side = side
                inner = [[positions[i, 0], positions[i, 1], np.degrees(angles[i])] for i in range(N)]
                best_result = (inner, (outer_center[0], outer_center[1]), side, np.degrees(outer_angle))
                
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 50.0, 0.0
    
    return best_result
