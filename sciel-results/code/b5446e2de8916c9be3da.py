import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # 1. Define the rigid cluster of 12 hexagons
    # Using axial coordinates (q, r) for a honeycomb packing
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (partial): (2,0), (1,1), (0,2), (-1,2), (-2,1)
    # This forms a compact cluster of 12 hexagons.
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    # Convert axial to Cartesian for unit hexagons (side length 1)
    # Distance between centers is sqrt(3)
    sqrt3 = np.sqrt(3.0)
    centers = np.zeros((N, 2))
    for i, (q, r) in enumerate(coords):
        centers[i, 0] = sqrt3 * (q + r / 2.0)
        centers[i, 1] = 1.5 * r
        
    # Precompute vertices for each hexagon in the cluster (local coordinates)
    # Hexagon vertices at angles 0, 60, ..., 300
    angles = np.radians(np.arange(0, 360, 60))
    local_verts = np.column_stack([np.cos(angles), np.sin(angles)])
    # local_verts shape: (6, 2)
    
    # 2. Define the objective function: outer side length for a given rotation
    def required_outer_side(rotation_rad):
        """
        Compute the minimum side length of the outer hexagon (centered at origin, fixed orientation)
        that contains the rotated cluster.
        """
        # Rotate centers
        c = np.cos(rotation_rad)
        s = np.sin(rotation_rad)
        R = np.array([[c, -s], [s, c]])
        
        rotated_centers = centers @ R.T
        
        # Rotate local vertices
        rotated_local_verts = local_verts @ R.T
        
        # Translate and stack all vertices
        # verts[i] = rotated_centers[i] + rotated_local_verts
        # We need to find the max projection onto the 6 normal directions of the outer hexagon.
        
        # Normals for outer hexagon (side S, centered at origin, vertices at 0, 60...):
        # The sides are perpendicular to normals at 30, 90, 150, 210, 270, 330.
        # n_k = (cos(30 + 60k), sin(30 + 60k))
        # The condition is: p . n_k <= S * sqrt(3) / 2
        # So S >= (2 / sqrt(3)) * max(p . n_k)
        
        normal_angles = np.radians(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Project all vertices onto all normals
        # To be efficient, we can project the centers and local verts separately
        # p = c_i + v_j
        # p . n = c_i . n + v_j . n
        
        # c_i . n for all i and all 6 normals
        center_projs = rotated_centers @ normals.T  # (N, 6)
        
        # v_j . n for all 6 local verts and all 6 normals
        local_projs = rotated_local_verts @ normals.T  # (6, 6)
        
        # Max projection for each normal k is max_i (center_projs[i,k] + max_j local_projs[j,k])
        # Since local_projs is the same for all i, we can take max over j first
        max_local_proj = np.max(local_projs, axis=0)  # (6,)
        
        # Now for each normal k, we need max_i (center_projs[i,k]) + max_local_proj[k]
        max_center_proj = np.max(center_projs, axis=0)  # (6,)
        
        max_projs = max_center_proj + max_local_proj
        
        max_proj = np.max(max_projs)
        
        return (2.0 / sqrt3) * max_proj

    # 3. Grid search over rotation
    # The system has 6-fold symmetry, so we only need to search 0 to 60 degrees.
    # Let's do a fine grid search.
    # 3600 steps in 60 degrees -> 0.016 degree resolution.
    num_steps = 3600
    angles_rad = np.linspace(0, np.pi / 3, num_steps)
    
    best_side = float('inf')
    best_angle = 0.0
    
    # Vectorized evaluation if possible, or loop
    # Since required_outer_side uses loops internally (matmul), it's fast enough.
    for ang in angles_rad:
        if time.time() > deadline:
            break
        side = required_outer_side(ang)
        if side < best_side:
            best_side = side
            best_angle = ang
            
    # Refine around the best angle
    if time.time() < deadline - 10:
        refine_range = 0.01
        refine_steps = 1000
        refine_angles = np.linspace(best_angle - refine_range, best_angle + refine_range, refine_steps)
        for ang in refine_angles:
            side = required_outer_side(ang)
            if side < best_side:
                best_side = side
                best_angle = ang

    # 4. Construct the output
    # Inner hexagons are placed at the rotated centers, with rotation best_angle
    c = np.cos(best_angle)
    s = np.sin(best_angle)
    R = np.array([[c, -s], [s, c]])
    final_centers = centers @ R.T
    
    inner = []
    for i in range(N):
        inner.append([float(final_centers[i, 0]), float(final_centers[i, 1]), np.degrees(best_angle)])
        
    # Outer hexagon: center (0,0), side best_side, rotation 0
    return inner, (0.0, 0.0), float(best_side), 0.0
