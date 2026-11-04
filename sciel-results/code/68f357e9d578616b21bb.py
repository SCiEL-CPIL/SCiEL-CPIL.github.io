import numpy as np
import time

def construct():
    # Deadline guard: ensure we finish well within 150s
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin

    # Strategy: Concentric ring packing with 11 hexagons
    # Ring 0: 1 hexagon at center
    # Ring 1: 6 hexagons at distance sqrt(3) (touching neighbors)
    # Ring 2: 4 hexagons at distance 3 (filling gaps in ring 1)
    
    # Use a small epsilon to ensure strict disjointness (interiors don't overlap)
    # The minimum distance between centers of touching unit hexagons is sqrt(3)
    # We add eps to ensure a tiny gap
    eps = 1e-7

    centers = []

    # Ring 0: center
    centers.append((0.0, 0.0))

    # Ring 1: 6 hexagons at distance sqrt(3) + eps, angles 0, 60, 120, 180, 240, 300
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))

    # Ring 2: 4 hexagons at distance 3 + 2*eps, angles 30, 90, 150, 270
    # Note: The original code used 30, 90, 150, 210 (k=0..3).
    # Let's stick to the original code's positions to ensure validity, 
    # but verify if 270 is better? 
    # Original: 30, 90, 150, 210.
    # If we use 30, 90, 150, 270, we have 4 hexes.
    # 210 and 270 are different.
    # Let's check the original code again.
    # "for k in range(4): angle = np.deg2rad(30 + k * 60)" -> 30, 90, 150, 210.
    # I will keep the original positions to be safe, as they are known to be disjoint.
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))

    # Precompute vertices of a unit hexagon for angle 0 (Pointy-Right)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    # Wait, if angle=0 is Pointy-Right, vertices are at 0, 60...
    # If we want to rotate by phi_in, we add phi_in to these angles.
    # Let's define the base vertices for angle 0.
    base_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    base_vertices = np.column_stack([np.cos(base_angles), np.sin(base_angles)])

    # Search for optimal rotations
    # phi_in: rotation of inner hexagons
    # phi_out: rotation of outer hexagon
    # We expect the optimal to be near 30 degrees (Pointy-Top)
    
    best_side = float('inf')
    best_phi_in = 30.0
    best_phi_out = 30.0
    
    # Grid search
    # Resolution: 1 degree
    # Range: 0 to 60 (symmetry)
    phi_in_vals = np.arange(0, 61, 1.0)
    phi_out_vals = np.arange(0, 61, 1.0)
    
    # Optimization: Precompute centers as array
    centers_arr = np.array(centers)
    
    for phi_in in phi_in_vals:
        # Check time
        if time.time() - start_time > DEADLINE - 10:
            break
            
        rad_in = np.deg2rad(phi_in)
        # Rotate base vertices
        cos_in = np.cos(rad_in)
        sin_in = np.sin(rad_in)
        
        # Rotation matrix
        R_in = np.array([[cos_in, -sin_in], [sin_in, cos_in]])
        vertices_in = base_vertices @ R_in.T
        
        for phi_out in phi_out_vals:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            rad_out = np.deg2rad(phi_out)
            # Normals of the outer hexagon
            # Sides are perpendicular to angles phi_out + 30, phi_out + 90, etc?
            # For a hexagon rotated by phi_out (where phi_out is the angle of the first vertex? No, let's define phi_out as the angle of the "top" vertex or similar).
            # Let's use the normal angles.
            # If the hexagon is "Pointy-Top" (vertex at 90), the normals are at 0, 60, 120, 180, 240, 300.
            # If we rotate the hexagon by phi_out, the normals are at phi_out, phi_out+60, ...
            # Wait, if phi_out=30 corresponds to Pointy-Top in our previous analysis?
            # Previous analysis: Angle 30 -> Vertices at 30, 90... -> Pointy Top.
            # Normals for Pointy Top (vertex at 90) are at 0, 60, 120, 180, 240, 300.
            # So if phi_out=30, normals should be at 0, 60...
            # So normal_angle = phi_out - 30 + k*60.
            
            normal_angles = np.deg2rad(phi_out - 30 + np.array([0, 60, 120, 180, 240, 300]))
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
            
            # Calculate max projection for all vertices of all inner hexagons
            # Vertices of inner hex i: centers[i] + vertices_in
            # We need max over all these vertices of dot(p, n)
            
            # Vectorized calculation:
            # p = centers[:, np.newaxis, :] + vertices_in[np.newaxis, :, :]
            # Shape: (N_centers, 6_vertices, 2)
            # Normals: (6_normals, 2)
            # Dot product: (N_centers, 6_vertices, 6_normals)
            
            # To save memory/time, we can loop over centers or use broadcasting carefully.
            # N_centers = 11. 6 vertices. 6 normals. 11*6*6 = 396 operations. Very small.
            
            p = centers_arr[:, np.newaxis, :] + vertices_in[np.newaxis, :, :]
            # p shape: (11, 6, 2)
            # normals shape: (6, 2)
            
            # dot(p, normals) -> (11, 6, 6)
            dots = np.einsum('ijk,lk->ijl', p, normals)
            
            # max over vertices (axis 1) and normals (axis 2)
            max_proj = np.max(dots)
            
            # S = 2/sqrt(3) * max_proj
            side = (2.0 / np.sqrt(3.0)) * max_proj
            
            if side < best_side:
                best_side = side
                best_phi_in = phi_in
                best_phi_out = phi_out

    # Add tiny margin
    best_side += 1e-6
    
    # Construct inner list with best_phi_in
    inner = []
    for x, y in centers:
        inner.append([x, y, best_phi_in])

    return inner, (0.0, 0.0), best_side, best_phi_out
