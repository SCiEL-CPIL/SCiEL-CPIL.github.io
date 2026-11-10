# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _get_all_vertices(positions, angles, side=1.0):
    """positions: (N,2), angles: (N,) in degrees. Returns (N*6, 2)."""
    verts = []
    for i in range(len(positions)):
        verts.append(_hex_vertices(positions[i,0], positions[i,1], side, angles[i]))
    return np.vstack(verts)

def _min_outer_side(vertices):
    """
    Compute minimum side length s of a regular hexagon centered at origin (angle 0)
    that contains all given vertices.
    For a regular hex with side s and angle 0, vertices are at s*(cos(60k), sin(60k)).
    The 6 inward-facing normals to the edges point at angles 30, 90, 150, 210, 270, 330 degrees.
    The distance from center to each edge (apothem) is s * sqrt(3)/2.
    So s_min = (2/sqrt(3)) * max over all vertices v of max over 6 inward normals n of (n . v)
    """
    # Inward normals for edges of hex with vertices at 0,60,...,300 deg
    # Edge between vertex at 0 and 60 deg: midpoint direction is 30 deg, inward normal is 30 deg
    # Actually, let's think carefully.
    # Vertex k is at angle 60k. Edge k is between vertex k and vertex k+1.
    # The outward normal to edge k points at angle 60k + 30.
    # So inward normal points at 60k + 30 + 180 = 60k + 210, or equivalently 60k - 150.
    # Let's just use: for a hex centered at origin with vertices at radius s at angles 60k,
    # the half-width in direction theta is s * cos(30) / cos(theta - 60*round(theta/60)) ... 
    # Simpler: the apothem (distance from center to edge) is s*sqrt(3)/2.
    # The 6 edge normals (outward) are at angles 30, 90, 150, 210, 270, 330 degrees.
    # For a point v to be inside, we need v . n_out <= apothem for all 6 outward normals.
    # So apothem_min = max over v of max over 6 outward normals of (v . n_out)
    # s_min = 2 * apothem_min / sqrt(3)
    
    outward_normals = np.array([
        [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))]
        for k in range(6)
    ])  # (6,2)
    
    # vertices: (M, 2)
    # For each vertex and each normal, compute dot product
    # dots[i, j] = vertices[i] . outward_normals[j]
    dots = vertices @ outward_normals.T  # (M, 6)
    max_dot = np.max(dots, axis=1)  # (M,)
    apothem_min = np.max(max_dot)
    s_min = 2.0 * apothem_min / np.sqrt(3.0)
    return s_min

def _check_separation(pos1, ang1, pos2, ang2, side=1.0):
    """
    Check if two unit hexagons are separated using SAT.
    Returns True if separated (no overlap), False if overlapping.
    """
    v1 = _hex_vertices(pos1[0], pos1[1], side, ang1)  # (6,2)
    v2 = _hex_vertices(pos2[0], pos2[1], side, ang2)  # (6,2)
    
    # Get all edge normals from both hexagons (12 axes)
    normals = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i+1)%6] - v[i]
            # Normal to edge
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                normals.append(n)
    
    for n in normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        # If intervals don't overlap, separated
        if max1 < min2 - 1e-8 or max2 < min1 - 1e-8:
            return True
    
    return False

def _overlap_penalty(pos, ang, side=1.0, eps=1e-6):
    """
    Compute a penalty for overlaps between all pairs of hexagons.
    Uses a soft measure: for each pair, compute minimum distance between edges/vertices.
    Simpler: use SAT-based penetration depth approximation.
    """
    penalty = 0.0
    n = len(pos)
    for i in range(n):
        for j in range(i+1, n):
            # Quick rejection by center distance
            d = np.linalg.norm(pos[i] - pos[j])
            if d > 2.0 * side + 1e-3:
                continue
            # Compute overlap using SAT
            v1 = _hex_vertices(pos[i,0], pos[i,1], side, ang[i])
            v2 = _hex_vertices(pos[j,0], pos[j,1], side, ang[j])
            
            normals = []
            for v in [v1, v2]:
                for k in range(6):
                    edge = v[(k+1)%6] - v[k]
                    nn = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(nn)
                    if norm > 1e-12:
                        nn = nn / norm
                        normals.append(nn)
            
            min_overlap = np.inf
            for nrm in normals:
                proj1 = v1 @ nrm
                proj2 = v2 @ nrm
                overlap = min(np.max(proj1), np.max(proj2)) - max(np.min(proj1), np.min(proj2))
                if overlap > 0:
                    min_overlap = min(min_overlap, overlap)
            
            if min_overlap < np.inf:
                penalty += min_overlap ** 2
    
    return penalty

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, leave margin
    
    n = 12
    side = 1.0
    
    # Initial configuration: hexagonal grid pattern
    # Centers at triangular lattice positions
    dx = 1.5  # horizontal spacing for unit hexagons (edge to edge)
    dy = np.sqrt(3)  # vertical spacing
    
    # 12 hexagons in a compact arrangement
    # Row 0: 3 hexagons
    # Row 1: 3 hexagons offset
    # Row 2: 3 hexagons
    # Row 3: 3 hexagons offset
    # Actually, let's use a 4x3 grid
    init_positions = []
    init_angles = []
    for row in range(4):
        for col in range(3):
            x = col * 2 * side * 1.5
            y = row * side * np.sqrt(3)
            if row % 2 == 1:
                x += side * 1.5
            init_positions.append([x, y])
            init_angles.append(0.0)
    
    init_positions = np.array(init_positions)
    init_angles = np.array(init_angles)
    
    # Center the configuration
    centroid = np.mean(init_positions, axis=0)
    init_positions -= centroid
    
    best_pos = init_positions.copy()
    best_ang = init_angles.copy()
    best_s = _min_outer_side(_get_all_vertices(best_pos, best_ang, side))
    
    # Gradient-based optimization
    lr = 0.05
    step = 0
    
    while time.time() < deadline - 30.0:
        pos = best_pos.copy()
        ang = best_ang.copy()
        
        # Compute objective: s_min + overlap_penalty
        vertices = _get_all_vertices(pos, ang, side)
        s = _min_outer_side(vertices)
        overlap_pen = _overlap_penalty(pos, ang, side)
        objective = s + 10.0 * overlap_pen
        
        # Compute gradients numerically (or analytically for s)
        # For s_min: gradient w.r.t. vertex positions
        # s = 2/sqrt(3) * max_{v, n} (v . n)
        # The active constraint is the vertex/normal pair giving max dot
        outward_normals = np.array([
            [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))]
            for k in range(6)
        ])
        dots = vertices @ outward_normals.T
        max_dot = np.max(dots, axis=1)
        active_idx = np.argmax(max_dot)
        active_normal_idx = np.argmax(dots[active_idx])
        active_normal = outward_normals[active_normal_idx]
        
        # Gradient of s w.r.t. vertex positions: 2/sqrt(3) * active_normal for active vertex
        # Each vertex belongs to a hexagon
        hex_idx = active_idx // 6
        vert_idx = active_idx % 6
        
        # Gradient of s w.r.t. pos[hex_idx]
        grad_s_pos = np.zeros((n, 2))
        grad_s_pos[hex_idx] = (2.0 / np.sqrt(3.0)) * active_normal
        
        # Gradient w.r.t. angle: vertex = center + side*(cos(a+60k), sin(a+60k))
        # d(vertex)/d(a) = side*(-sin(a+60k), cos(a+60k))
        a_rad = np.radians(ang[hex_idx])
        vert_angle = a_rad + vert_idx * np.pi / 3
        dvert_dang = side * np.array([-np.sin(vert_angle), np.cos(vert_angle)])
        grad_s_ang = np.zeros(n)
        grad_s_ang[hex_idx] = (2.0 / np.sqrt(3.0)) * (active_normal @ dvert_dang)
        
        # Gradient of overlap penalty: numerical
        eps = 1e-4
        grad_pen_pos = np.zeros((n, 2))
        grad_pen_ang = np.zeros(n)
        
        # Only compute gradient for pairs that might be close
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d > 2.0 * side + 1e-2:
                    continue
                # Perturb pos[i]
                for dim in range(2):
                    pos_p = pos.copy()
                    pos_m = pos.copy()
                    pos_p[i, dim] += eps
                    pos_m[i, dim] -= eps
                    pen_p = _overlap_penalty(pos_p, ang, side)
                    pen_m = _overlap_penalty(pos_m, ang, side)
                    grad_pen_pos[i, dim] += (pen_p - pen_m) / (2 * eps)
                
                # Perturb pos[j]
                for dim in range(2):
                    pos_p = pos.copy()
                    pos_m = pos.copy()
                    pos_p[j, dim] += eps
                    pos_m[j, dim] -= eps
                    pen_p = _overlap_penalty(pos_p, ang, side)
                    pen_m = _overlap_penalty(pos_m, ang, side)
                    grad_pen_pos[j, dim] += (pen_p - pen_m) / (2 * eps)
                
                # Perturb ang[i]
                ang_p = ang.copy()
                ang_m = ang.copy()
                ang_p[i] += eps
                ang_m[i] -= eps
                pen_p = _overlap_penalty(pos, ang_p, side)
                pen_m = _overlap_penalty(pos, ang_m, side)
                grad_pen_ang[i] += (pen_p - pen_m) / (2 * eps)
                
                # Perturb ang[j]
                ang_p = ang.copy()
                ang_m = ang.copy()
                ang_p[j] += eps
                ang_m[j] -= eps
                pen_p = _overlap_penalty(pos, ang_p, side)
                pen_m = _overlap_penalty(pos, ang_m, side)
                grad_pen_ang[j] += (pen_p - pen_m) / (2 * eps)
        
        grad_pos = grad_s_pos + 10.0 * grad_pen_pos
        grad_ang = grad_s_ang + 10.0 * grad_pen_ang
        
        # Update
        pos_new = pos - lr * grad_pos
        ang_new = ang - lr * grad_ang
        
        # Evaluate new configuration
        vertices_new = _get_all_vertices(pos_new, ang_new, side)
        s_new = _min_outer_side(vertices_new)
        overlap_new = _overlap_penalty(pos_new, ang_new, side)
        objective_new = s_new + 10.0 * overlap_new
        
        # Check if new is better
        if objective_new < objective:
            # Verify separation
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    if not _check_separation(pos_new[i], ang_new[i], pos_new[j], ang_new[j], side):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and s_new < best_s:
                best_pos = pos_new.copy()
                best_ang = ang_new.copy()
                best_s = s_new
        
        # Adaptive learning rate
        if objective_new >= objective:
            lr *= 0.95
        else:
            lr = min(lr * 1.05, 0.1)
        
        step += 1
        if step % 100 == 0:
            elapsed = time.time() - start_time
            if elapsed > deadline - 60.0:
                break
    
    # Final validation and return
    inner_hex_data = np.column_stack([best_pos, best_ang])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
