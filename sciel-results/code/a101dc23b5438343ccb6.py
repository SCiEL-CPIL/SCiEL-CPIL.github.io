# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.pi / 3 * np.arange(6)
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _sat_overlap(v1, v2):
    """Check if two convex polygons (arrays of vertices) overlap using SAT."""
    polys = [v1, v2]
    for i, poly in enumerate(polys):
        other = polys[1 - i]
        n = len(poly)
        for j in range(n):
            p1 = poly[j]
            p2 = poly[(j + 1) % n]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            
            proj1 = np.dot(poly, normal)
            proj2 = np.dot(other, normal)
            
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            if max1 < min2 or max2 < min1:
                return False
    return True

def _min_translation_to_separate(v1, v2):
    """Find minimum translation to separate two convex polygons."""
    best_dist = float('inf')
    best_normal = None
    polys = [v1, v2]
    for i, poly in enumerate(polys):
        other = polys[1 - i]
        n = len(poly)
        for j in range(n):
            p1 = poly[j]
            p2 = poly[(j + 1) % n]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            
            proj1 = np.dot(poly, normal)
            proj2 = np.dot(other, normal)
            
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            if max1 < min2:
                dist = min2 - max1
                if dist < best_dist:
                    best_dist = dist
                    best_normal = -normal
            elif max2 < min1:
                dist = min1 - max2
                if dist < best_dist:
                    best_dist = dist
                    best_normal = normal
    if best_normal is None:
        return np.array([0.0, 0.0])
    return best_dist * best_normal

def _calc_min_outer_hexagon(points):
    """
    Calculate the minimum side length of a regular hexagon enclosing the given points.
    Returns (side_length, center_x, center_y, angle_deg).
    """
    min_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for angle_deg in np.linspace(0, 60, 61):
        angle_rad = np.radians(angle_deg)
        
        normals = []
        for k in range(3):
            n_ang = angle_rad + np.radians(30) + np.radians(60) * k
            normals.append([np.cos(n_ang), np.sin(n_ang)])
        
        normals = np.array(normals)
        
        projections = np.dot(points, normals.T)
        
        max_range = 0
        center_proj = np.zeros(3)
        for k in range(3):
            min_p = projections[:, k].min()
            max_p = projections[:, k].max()
            range_k = max_p - min_p
            if range_k > max_range:
                max_range = range_k
            center_proj[k] = (min_p + max_p) / 2.0
        
        apothem = max_range / 2.0
        side = 2.0 * apothem / np.sqrt(3)
        
        if side < min_side:
            min_side = side
            best_angle = angle_deg
            A_mat = normals[:2].T
            b_vec = center_proj[:2]
            try:
                center = np.linalg.solve(A_mat, b_vec)
                best_center = (center[0], center[1])
            except np.linalg.LinAlgError:
                best_center = (0.0, 0.0)
                
    return min_side, best_center[0], best_center[1], best_angle

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    N = 11
    s = 1.0
    
    positions = np.array([[0.0, 0.0]])
    angles = np.array([0.0])
    
    for i in range(6):
        ang = np.pi / 3 * i
        r = np.sqrt(3) * s
        positions = np.vstack([positions, [r * np.cos(ang), r * np.sin(ang)]])
        angles = np.append(angles, 0.0)
        
    for i in range(4):
        ang = np.pi / 2 * i
        r = 2 * np.sqrt(3) * s
        positions = np.vstack([positions, [r * np.cos(ang), r * np.sin(ang)]])
        angles = np.append(angles, 0.0)
        
    assert len(positions) == N
    
    velocities = np.zeros_like(positions)
    ang_velocities = np.zeros(N)
    
    best_score = float('inf')
    best_result = None
    
    dt = 0.05
    damping = 0.9
    
    for step in range(2000):
        if time.time() > deadline:
            break
            
        forces = np.zeros_like(positions)
        torques = np.zeros(N)
        
        all_vertices = []
        for i in range(N):
            vi = _hex_vertices(positions[i][0], positions[i][1], s, angles[i])
            all_vertices.append(vi)
        all_vertices = np.vstack(all_vertices)
        
        overlap_count = 0
        for i in range(N):
            vi = all_vertices[i*6:(i+1)*6]
            for j in range(i+1, N):
                vj = all_vertices[j*6:(j+1)*6]
                if _sat_overlap(vi, vj):
                    overlap_count += 1
                    mt = _min_translation_to_separate(vi, vj)
                    forces[i] += mt * 10.0
                    forces[j] -= mt * 10.0
                    r_vec = positions[i] - positions[j]
                    torque = r_vec[0] * mt[1] - r_vec[1] * mt[0]
                    torques[i] += 0.5 * torque
                    torques[j] -= 0.5 * torque
        
        for i in range(N):
            forces[i] -= positions[i] * 0.5
            
        velocities = velocities * damping + forces * dt
        positions = positions + velocities * dt
        ang_velocities = ang_velocities * damping + torques * dt
        angles = angles + ang_velocities * dt
        angles = np.mod(angles, 2 * np.pi)
        
        if overlap_count == 0:
            side, cx, cy, ang = _calc_min_outer_hexagon(all_vertices)
            score = side
            if score < best_score:
                best_score = score
                best_result = (
                    [[float(positions[i][0]), float(positions[i][1]), float(np.degrees(angles[i]))] for i in range(N)],
                    (float(cx), float(cy)),
                    float(side),
                    float(ang)
                )
        
        if overlap_count == 0 and np.max(np.abs(velocities)) < 1e-4:
            break
            
    if best_result is not None:
        return best_result
    else:
        scale = 1.0
        while True:
            test_positions = positions * scale
            all_vertices = []
            for i in range(N):
                vi = _hex_vertices(test_positions[i][0], test_positions[i][1], s, angles[i])
                all_vertices.append(vi)
            all_vertices = np.vstack(all_vertices)
            overlap = False
            for i in range(N):
                vi = all_vertices[i*6:(i+1)*6]
                for j in range(i+1, N):
                    vj = all_vertices[j*6:(j+1)*6]
                    if _sat_overlap(vi, vj):
                        overlap = True
                        break
                if overlap:
                    break
            if not overlap:
                side, cx, cy, ang = _calc_min_outer_hexagon(all_vertices)
                return (
                    [[float(test_positions[i][0]), float(test_positions[i][1]), float(np.degrees(angles[i]))] for i in range(N)],
                    (float(cx), float(cy)),
                    float(side),
                    float(ang)
                )
            scale += 0.1
            if scale > 10:
                return (
                    [[float(test_positions[i][0]), float(test_positions[i][1]), float(np.degrees(angles[i]))] for i in range(N)],
                    (0.0, 0.0),
                    50.0,
                    0.0
                )
# EVOLVE-BLOCK-END
