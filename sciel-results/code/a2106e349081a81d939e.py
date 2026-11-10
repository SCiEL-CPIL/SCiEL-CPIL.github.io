# EVOLVE-BLOCK-START
import numpy as np
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600.0

    # Unit hexagon vertices (angle 0)
    # Vertices at 0, 60, ... 300 degrees
    hex_verts = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)
    ])
    
    # Triangular lattice vectors for ring 1
    # Distance sqrt(3)
    d = np.sqrt(3)
    v = []
    for i in range(6):
        angle = np.deg2rad(60 * i)
        v.append([d * np.cos(angle), d * np.sin(angle)])
    v = np.array(v)
    
    # Ring 2 vectors
    # 2*v[i] and v[i] + v[i+1]
    ring2 = []
    for i in range(6):
        ring2.append(2 * v[i])
        ring2.append(v[i] + v[(i + 1) % 6])
    ring2 = np.array(ring2)
    
    # Base centers: Center + Ring 1
    base_centers = np.vstack([np.array([[0, 0]]), v])
    
    # Candidate sets of 4 ring-2 points
    # We want to try symmetric sets.
    # Angles of ring2 points:
    # 2*v[i] are at 0, 60, 120, 180, 240, 300
    # v[i]+v[i+1] are at 30, 90, 150, 210, 270, 330
    
    # Let's index them:
    # 0: 2v0 (0 deg)
    # 1: v0+v1 (30 deg)
    # 2: 2v1 (60 deg)
    # 3: v1+v2 (90 deg)
    # 4: 2v2 (120 deg)
    # 5: v2+v3 (150 deg)
    # 6: 2v3 (180 deg)
    # 7: v3+v4 (210 deg)
    # 8: 2v4 (240 deg)
    # 9: v4+v5 (270 deg)
    # 10: 2v5 (300 deg)
    # 11: v5+v0 (330 deg)
    
    # Try a few symmetric combinations of 4 indices
    candidates = [
        [0, 3, 6, 9],    # 0, 90, 180, 270
        [0, 2, 6, 8],    # 0, 60, 180, 240
        [1, 4, 7, 10],   # 30, 120, 210, 300
        [0, 5, 6, 11],   # 0, 150, 180, 330
        [1, 3, 7, 9],    # 30, 90, 210, 270
        [2, 4, 8, 10],   # 60, 120, 240, 300
    ]
    
    best_side = float('inf')
    best_config = None
    
    for idxs in candidates:
        if time.time() > deadline - 10:
            break
            
        r2_pts = ring2[idxs]
        centers = np.vstack([base_centers, r2_pts])
        
        # Calculate the "obstacle" hull
        # The obstacle is the Minkowski sum of the set of centers and the unit hexagon.
        # Since we only need the support function in 6 directions (outer normals),
        # we can compute the max projection of (center + hex_vert) for each direction.
        
        # To optimize the outer angle, we scan angles.
        # The function S(theta) is piecewise linear-ish, we can sample it.
        
        # Outer hexagon normals are at theta, theta+60, ...
        # We want to minimize max_i h(theta + 60*i)
        
        # Let's sample theta from 0 to 60 degrees with fine steps.
        thetas = np.linspace(0, np.pi/3, 360)
        
        min_side_for_this_cluster = float('inf')
        best_theta = 0
        
        for theta in thetas:
            # Normals
            normals = []
            for i in range(6):
                a = theta + np.deg2rad(60 * i)
                normals.append([np.cos(a), np.sin(a)])
            normals = np.array(normals)
            
            # Compute support function for each normal
            # h(n) = max_{c in centers, v in hex_verts} n . (c + v)
            # = max_c (n . c) + max_v (n . v)
            
            # max_v (n . v) is the apothem of the unit hexagon if n is aligned with a normal?
            # No, it's the support function of the unit hexagon in direction n.
            # For a regular hexagon of side 1, the support function h(n) depends on the angle of n relative to the hexagon.
            # If n is a normal to an edge, h(n) = apothem = sqrt(3)/2.
            # If n is towards a vertex, h(n) = 1.
            # It varies between sqrt(3)/2 and 1.
            
            # Let's compute it exactly for each normal.
            # n is unit vector.
            # The vertices of the unit hexagon are at angles 0, 60, ...
            # h_hex(n) = max_i (n . hex_verts[i])
            
            # We can precompute hex_verts
            # h_hex = np.max(normals @ hex_verts.T, axis=1)
            
            # h_total = h_centers + h_hex
            # h_centers = max_c (n . c)
            
            # S = (2/sqrt(3)) * max(h_total)
            
            # Vectorized:
            # normals: (6, 2)
            # centers: (11, 2)
            # hex_verts: (6, 2)
            
            # h_centers: (6,)
            # proj_centers = centers @ normals.T  # (11, 6)
            # max_proj_centers = np.max(proj_centers, axis=0)
            
            # h_hex: (6,)
            # proj_hex = hex_verts @ normals.T  # (6, 6)
            # max_proj_hex = np.max(proj_hex, axis=0)
            
            # h_total = max_proj_centers + max_proj_hex
            # max_h = np.max(h_total)
            # side = (2 / np.sqrt(3)) * max_h
            
            # Check time
            if time.time() > deadline - 5:
                if best_config is not None:
                    return best_config[0], best_config[1], best_config[2]
                # Fallback
                return _fallback()
                
            proj_centers = centers @ normals.T
            max_proj_centers = np.max(proj_centers, axis=0)
            
            proj_hex = hex_verts @ normals.T
            max_proj_hex = np.max(proj_hex, axis=0)
            
            h_total = max_proj_centers + max_proj_hex
            max_h = np.max(h_total)
            
            side = (2.0 / np.sqrt(3.0)) * max_h
            
            if side < min_side_for_this_cluster:
                min_side_for_this_cluster = side
                best_theta = theta
                
        if min_side_for_this_cluster < best_side:
            best_side = min_side_for_this_cluster
            # Store the configuration
            # inner: centers + angle 0
            inner_hex_data = np.column_stack([centers, np.zeros(11)])
            # outer: center 0,0, angle best_theta
            outer_hex_data = np.array([0, 0, np.rad2deg(best_theta)])
            best_config = (inner_hex_data, outer_hex_data, best_side)
            
    if best_config is None:
        return _fallback()

    return best_config[0], best_config[1], best_config[2]

def _fallback():
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
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8.0
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
