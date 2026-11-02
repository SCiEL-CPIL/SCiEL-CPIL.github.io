"""Rigid honeycomb cluster: 12 hexagons in fixed 3x4 topology, optimize cluster
rotation and outer hexagon orientation via exhaustive angular search."""
import numpy as np
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def outer_side_fast(inner_hexes, outer_cx, outer_cy, outer_angle_rad):
        """Minimum outer hexagon side to contain all inner hexagons (support function)."""
        max_required = 0.0
        theta = outer_angle_rad
        for hex_verts in inner_hexes:
            for v in hex_verts:
                dx = v[0] - outer_cx
                dy = v[1] - outer_cy
                for k in range(6):
                    edge_angle = theta + np.radians(k * 60)
                    nx = np.cos(edge_angle)
                    ny = np.sin(edge_angle)
                    dist = dx * nx + dy * ny
                    if dist > 0:
                        required_S = dist * 2.0 / np.sqrt(3.0)
                        if required_S > max_required:
                            max_required = required_S
        return max_required

    def sat_disjoint(poly1, poly2, tol=1e-8):
        for poly in [poly1, poly2]:
            for i in range(6):
                v1 = poly[i]
                v2 = poly[(i + 1) % 6]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                proj1 = np.dot(poly1, normal)
                proj2 = np.dot(poly2, normal)
                if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                    return True
        return False

    # Define honeycomb topologies
    sqrt3 = np.sqrt(3)
    topologies = []
    
    # Topology 1: 3x4 staggered honeycomb (standard)
    positions1 = []
    for r in range(3):
        for c in range(4):
            x = c * sqrt3
            y = r * 1.5
            if r % 2 == 1:
                x += sqrt3 / 2
            positions1.append((x, y))
    topologies.append(np.array(positions1))
    
    # Topology 2: 4x3 staggered (transposed)
    positions2 = []
    for r in range(4):
        for c in range(3):
            x = c * sqrt3
            y = r * 1.5
            if r % 2 == 1:
                x += sqrt3 / 2
            positions2.append((x, y))
    topologies.append(np.array(positions2))
    
    # Topology 3: Compact hexagonal cluster (center + 5 around + 6 outer)
    positions3 = []
    positions3.append((0, 0))  # center
    for i in range(5):  # 5 around center (one missing for 12 total)
        angle = 2 * np.pi * i / 6
        positions3.append((sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)))
    for i in range(6):  # outer ring
        angle = 2 * np.pi * i / 6 + np.pi / 6
        r = 2 * sqrt3
        positions3.append((r * np.cos(angle), r * np.sin(angle)))
    topologies.append(np.array(positions3))
    
    # Topology 4: 2x6 grid
    positions4 = []
    for r in range(2):
        for c in range(6):
            x = c * sqrt3
            y = r * 1.5
            if r % 2 == 1:
                x += sqrt3 / 2
            positions4.append((x, y))
    topologies.append(np.array(positions4))
    
    # Center each topology
    for i, pos in enumerate(topologies):
        topologies[i] = pos - np.mean(pos, axis=0)

    best_side = np.inf
    best_result = None

    # For each topology, search over cluster rotation and outer hexagon orientation
    for topo_idx, topo in enumerate(topologies):
        if time.time() > deadline - 10:
            break
        
        # Coarse search: 1-degree steps for cluster rotation, 1-degree for outer
        for cluster_rot_deg in np.arange(0, 60, 1.0):
            if time.time() > deadline - 5:
                break
            
            cluster_rot = np.radians(cluster_rot_deg)
            rot_matrix = np.array([
                [np.cos(cluster_rot), -np.sin(cluster_rot)],
                [np.sin(cluster_rot), np.cos(cluster_rot)]
            ])
            
            # Rotate all positions
            rotated_topo = topo @ rot_matrix.T
            
            # Place hexagons at rotated positions with same rotation
            inner_hexes = []
            for i in range(N):
                verts = hexagon_vertices(rotated_topo[i, 0], rotated_topo[i, 1], 1.0, cluster_rot)
                inner_hexes.append(verts)
            
            # Check disjointness
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if not sat_disjoint(inner_hexes[i], inner_hexes[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Search over outer hexagon orientation
            for outer_rot_deg in np.arange(0, 60, 1.0):
                outer_rot = np.radians(outer_rot_deg)
                side = outer_side_fast(inner_hexes, 0.0, 0.0, outer_rot)
                
                if side < best_side:
                    best_side = side
                    best_result = (topo_idx, cluster_rot, outer_rot, inner_hexes.copy(), rotated_topo.copy())
        
        # Fine search around best found for this topology
        if best_result and best_result[0] == topo_idx:
            best_cluster_rot = best_result[1]
            best_outer_rot = best_result[2]
            
            for cluster_rot_deg in np.arange(best_cluster_rot - 1, best_cluster_rot + 1, 0.1):
                if time.time() > deadline - 3:
                    break
                
                cluster_rot = np.radians(cluster_rot_deg)
                rot_matrix = np.array([
                    [np.cos(cluster_rot), -np.sin(cluster_rot)],
                    [np.sin(cluster_rot), np.cos(cluster_rot)]
                ])
                
                rotated_topo = topo @ rot_matrix.T
                
                inner_hexes = []
                for i in range(N):
                    verts = hexagon_vertices(rotated_topo[i, 0], rotated_topo[i, 1], 1.0, cluster_rot)
                    inner_hexes.append(verts)
                
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        if not sat_disjoint(inner_hexes[i], inner_hexes[j]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                for outer_rot_deg in np.arange(best_outer_rot - 1, best_outer_rot + 1, 0.1):
                    outer_rot = np.radians(outer_rot_deg)
                    side = outer_side_fast(inner_hexes, 0.0, 0.0, outer_rot)
                    
                    if side < best_side:
                        best_side = side
                        best_result = (topo_idx, cluster_rot, outer_rot, inner_hexes.copy(), rotated_topo.copy())

    # If no valid result found, use a widely spaced fallback
    if best_result is None:
        inner = [[3.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 35.0, 0.0

    topo_idx, cluster_rot, outer_rot, inner_hexes, rotated_topo = best_result
    
    # Extract positions and angles
    inner_x = rotated_topo[:, 0]
    inner_y = rotated_topo[:, 1]
    inner_a = np.full(N, cluster_rot)
    
    # Recompute exact side length
    exact_side = outer_side_fast(inner_hexes, 0.0, 0.0, outer_rot)
    
    inner = [[inner_x[i], inner_y[i], np.degrees(inner_a[i])] for i in range(N)]
    return inner, (0.0, 0.0), exact_side, np.degrees(outer_rot)
# EVOLVE-BLOCK-END
