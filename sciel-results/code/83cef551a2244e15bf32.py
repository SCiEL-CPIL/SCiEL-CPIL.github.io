"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        apothem = side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        dists = local_x * nx + local_y * ny
        return np.all(dists <= apothem + 1e-9)
    
    def polygons_disjoint_sat(v1, v2):
        edges1 = np.roll(v1, -1, axis=0) - v1
        norms1 = np.column_stack([-edges1[:, 1], edges1[:, 0]])
        norms1 /= (np.linalg.norm(norms1, axis=1, keepdims=True) + 1e-9)
        
        edges2 = np.roll(v2, -1, axis=0) - v2
        norms2 = np.column_stack([-edges2[:, 1], edges2[:, 0]])
        norms2 /= (np.linalg.norm(norms2, axis=1, keepdims=True) + 1e-9)
        
        all_axes = np.vstack([norms1, norms2])
        
        proj1 = np.dot(v1, all_axes.T)
        proj2 = np.dot(v2, all_axes.T)
        
        min1 = np.min(proj1, axis=0)
        max1 = np.max(proj1, axis=0)
        min2 = np.min(proj2, axis=0)
        max2 = np.max(proj2, axis=0)
        
        separated = (max1 < min2 - 1e-9) | (max2 < min1 - 1e-9)
        return np.any(separated)
    
    def hexagons_disjoint(v1, s1, a1, v2, s2, a2):
        verts1 = hexagon_vertices(v1[0], v1[1], s1, a1)
        verts2 = hexagon_vertices(v2[0], v2[1], s2, a2)
        return polygons_disjoint_sat(verts1, verts2)
    
    def all_inside_outer(inner_positions, outer_center, outer_side, outer_angle):
        n_inner = len(inner_positions)
        if n_inner == 0: return True
        
        centers = np.array([(p[0], p[1]) for p in inner_positions])
        angles = np.array([p[2] for p in inner_positions])
        
        angle_rad = np.radians(angles)
        base_angles = np.radians(np.arange(0, 360, 60))
        
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        
        local_verts = np.column_stack([
            np.cos(base_angles),
            np.sin(base_angles)
        ])
        
        rx = local_verts[:, 0]
        ry = local_verts[:, 1]
        
        rotated_x = rx * cos_a[:, np.newaxis] - ry * sin_a[:, np.newaxis]
        rotated_y = rx * sin_a[:, np.newaxis] + ry * cos_a[:, np.newaxis]
        
        verts_x = rotated_x + centers[:, 0:1]
        verts_y = rotated_y + centers[:, 1:2]
        
        all_verts = np.stack([verts_x, verts_y], axis=-1)
        
        flat_verts = all_verts.reshape(-1, 2)
        
        ocx, ocy = outer_center
        dx = flat_verts[:, 0] - ocx
        dy = flat_verts[:, 1] - ocy
        
        outer_angle_rad = np.radians(outer_angle)
        cos_o = np.cos(-outer_angle_rad)
        sin_o = np.sin(-outer_angle_rad)
        
        local_x = dx * cos_o - dy * sin_o
        local_y = dx * sin_o + dy * cos_o
        
        apothem = outer_side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        dists = local_x[:, np.newaxis] * nx[np.newaxis, :] + local_y[:, np.newaxis] * ny[np.newaxis, :]
        
        if np.any(dists > apothem + 1e-9):
            return False
            
        return True
    
    def pairwise_disjoint(inner_positions):
        n_inner = len(inner_positions)
        for i in range(n_inner):
            for j in range(i+1, n_inner):
                ci = (inner_positions[i][0], inner_positions[i][1])
                cj = (inner_positions[j][0], inner_positions[j][1])
                ai = inner_positions[i][2]
                aj = inner_positions[j][2]
                if not hexagons_disjoint(ci, 1.0, ai, cj, 1.0, aj):
                    return False
        return True

    # Pre-validated dense honeycomb cluster of 12 unit hexagons
    # Using hexagonal lattice coordinates
    sqrt3 = np.sqrt(3)
    
    # Define cluster in local coordinates (before global rotation)
    # Hex grid with spacing sqrt(3) between centers
    # Pattern: center + 6 in ring 1 + 5 from ring 2 (to make 12)
    
    cluster_local = [
        (0, 0),           # center
        (sqrt3, 0),       # ring 1
        (-sqrt3, 0),
        (sqrt3/2, 1.5),
        (-sqrt3/2, 1.5),
        (sqrt3/2, -1.5),
        (-sqrt3/2, -1.5),
        (2*sqrt3, 0),     # ring 2 (partial)
        (sqrt3/2, 3.0),
        (-sqrt3/2, 3.0),
        (sqrt3/2, -3.0),
        (-sqrt3/2, -3.0),
    ]
    
    # Verify cluster is valid (all pairs disjoint)
    cluster_valid = True
    for i in range(len(cluster_local)):
        for j in range(i+1, len(cluster_local)):
            v1 = hexagon_vertices(cluster_local[i][0], cluster_local[i][1], 1.0, 0)
            v2 = hexagon_vertices(cluster_local[j][0], cluster_local[j][1], 1.0, 0)
            if not polygons_disjoint_sat(v1, v2):
                cluster_valid = False
                break
        if not cluster_valid:
            break
    
    if not cluster_valid:
        # Fallback: use a simpler grid if cluster is invalid
        cluster_local = []
        for i in range(3):
            y = i * sqrt3
            for j in range(4):
                x = (j - 1.5) * 2.0
                cluster_local.append((x, y))
    
    def required_outer_side(cluster_rotation, outer_rotation):
        """Compute minimum outer side length for given rotations."""
        # Rotate cluster
        rot_rad = np.radians(cluster_rotation)
        cos_r = np.cos(rot_rad)
        sin_r = np.sin(rot_rad)
        
        rotated_cluster = []
        for (x, y) in cluster_local:
            rx = x * cos_r - y * sin_r
            ry = x * sin_r + y * cos_r
            rotated_cluster.append((rx, ry))
        
        # Find bounding box of cluster
        xs = [p[0] for p in rotated_cluster]
        ys = [p[1] for p in rotated_cluster]
        
        # For a regular hexagon with side s and rotation theta,
        # the constraint is that all points must satisfy:
        # For each of 6 normal directions, dot(point - center, normal) <= s*sqrt(3)/2
        
        # Optimal center is centroid
        cx = np.mean(xs)
        cy = np.mean(ys)
        
        # Translate to center
        centered = [(x - cx, y - cy) for (x, y) in rotated_cluster]
        
        # For outer hexagon with rotation outer_rotation,
        # compute required side length
        outer_rot_rad = np.radians(outer_rotation)
        cos_o = np.cos(-outer_rot_rad)
        sin_o = np.sin(-outer_rot_rad)
        
        normal_angles = np.radians(np.arange(30, 360, 60))
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        max_dist = 0
        for (px, py) in centered:
            # Rotate point to outer hexagon's local frame
            lx = px * cos_o - py * sin_o
            ly = px * sin_o + py * cos_o
            dists = lx * nx + ly * ny
            max_dist = max(max_dist, np.max(dists))
        
        # Required side: apothem = s*sqrt(3)/2, so s = 2*apothem/sqrt(3)
        required_side = 2 * max_dist / sqrt3
        return required_side, (cx, cy)
    
    # Optimize over 2D: cluster_rotation, outer_rotation
    def objective_2d(params):
        cluster_rot = params[0]
        outer_rot = params[1]
        side, center = required_outer_side(cluster_rot, outer_rot)
        return side
    
    # Grid search over rotations
    best_side = float('inf')
    best_cluster_rot = 0
    best_outer_rot = 0
    best_center = (0, 0)
    
    # Coarse grid
    for c_rot in np.linspace(0, 60, 61):
        for o_rot in np.linspace(0, 60, 61):
            side, center = required_outer_side(c_rot, o_rot)
            if side < best_side:
                best_side = side
                best_cluster_rot = c_rot
                best_outer_rot = o_rot
                best_center = center
    
    # Fine optimization around best
    if time.time() < deadline - 10:
        try:
            res = minimize(objective_2d, [best_cluster_rot, best_outer_rot], 
                          method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True,
                                   'maxfev': 5000})
            if res.fun < best_side:
                best_side = res.fun
                best_cluster_rot = res.x[0]
                best_outer_rot = res.x[1]
                best_side, best_center = required_outer_side(best_cluster_rot, best_outer_rot)
        except Exception:
            pass
    
    # Very fine polish
    if time.time() < deadline - 5:
        try:
            res2 = minimize(objective_2d, [best_cluster_rot, best_outer_rot],
                           method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True,
                                    'maxfev': 2000})
            if res2.fun < best_side:
                best_side = res2.fun
                best_cluster_rot = res2.x[0]
                best_outer_rot = res2.x[1]
                best_side, best_center = required_outer_side(best_cluster_rot, best_outer_rot)
        except Exception:
            pass
    
    # Construct final configuration
    rot_rad = np.radians(best_cluster_rot)
    cos_r = np.cos(rot_rad)
    sin_r = np.sin(rot_rad)
    
    inner = []
    for (x, y) in cluster_local:
        rx = x * cos_r - y * sin_r
        ry = x * sin_r + y * cos_r
        # Translate to center
        rx += best_center[0]
        ry += best_center[1]
        inner.append([rx, ry, best_cluster_rot])
    
    ox, oy = best_center
    os = best_side
    oa = best_outer_rot
    
    # Final validation
    if not (all_inside_outer(inner, (ox, oy), os, oa) and pairwise_disjoint(inner)):
        # Expand outer side slightly if needed
        for _ in range(100):
            if all_inside_outer(inner, (ox, oy), os, oa) and pairwise_disjoint(inner):
                break
            os += 0.01
            if os > 15.0:
                break
    
    return inner, (ox, oy), os, oa
# EVOLVE-BLOCK-END
