import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = 150.0
    
    # Generate 3x4 block of hexagons
    # Flat-top orientation:
    # x spacing 1.5, y spacing sqrt(3), offset 1 for odd rows
    centers = []
    for j in range(3):
        for i in range(4):
            x = 1.5 * i + (1.0 if j % 2 == 1 else 0.0)
            y = np.sqrt(3) * j
            centers.append([x, y])
            
    centers = np.array(centers)
    # Center the cluster
    center_of_mass = np.mean(centers, axis=0)
    centers = centers - center_of_mass
    
    # Vertices of a unit flat-top hexagon (side 1)
    # Angles 0, 60, 120, 180, 240, 300
    angles = np.deg2rad(np.arange(0, 360, 60))
    unit_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_S = float('inf')
    best_theta_in = 0.0
    best_theta_out = 0.0
    best_inner = None
    
    # Search for optimal rotations
    # Symmetry of hexagon is 60 deg.
    # We can search 0 to 30 deg for both.
    theta_in_steps = np.linspace(0, np.pi/6, 31) # 0 to 30 deg
    theta_out_steps = np.linspace(0, np.pi/6, 31)
    
    for th_in in theta_in_steps:
        if time.time() > deadline - 10:
            break
        # Rotate inner centers
        c_in = np.array([[np.cos(th_in), -np.sin(th_in)], [np.sin(th_in), np.cos(th_in)]])
        rot_centers = centers @ c_in.T
        
        # Rotate unit vertices
        c_v = np.array([[np.cos(th_in), -np.sin(th_in)], [np.sin(th_in), np.cos(th_in)]])
        rot_verts = unit_verts @ c_v.T
        
        # Pre-compute all vertices of all inner hexagons
        # Shape: (12, 6, 2)
        all_verts = rot_centers[:, np.newaxis, :] + rot_verts[np.newaxis, :, :]
        
        for th_out in theta_out_steps:
            if time.time() > deadline - 5:
                break
            
            # Normals for outer hexagon
            # If outer hex is rotated by th_out, its normals are at 30 + th_out, 90 + th_out, etc.
            # Wait, standard flat-top hex has normals at 30, 90, ...
            # If we rotate the hex by th_out, the normals rotate by th_out.
            normal_angles = np.deg2rad(30) + th_out + np.arange(6) * np.pi/3
            
            # Calculate projection of all vertices onto normals
            # all_verts shape (72, 2)
            flat_verts = all_verts.reshape(-1, 2)
            
            # Projections: (72, 6)
            cos_n = np.cos(normal_angles)
            sin_n = np.sin(normal_angles)
            
            # P = X * cos + Y * sin
            proj_x = flat_verts[:, 0] * cos_n[np.newaxis, :]
            proj_y = flat_verts[:, 1] * sin_n[np.newaxis, :]
            proj = proj_x + proj_y
            
            max_proj = np.max(np.abs(proj))
            
            # Condition: max_proj <= Apothem = S * sqrt(3) / 2
            # S >= 2 * max_proj / sqrt(3)
            S_candidate = 2.0 * max_proj / np.sqrt(3)
            
            if S_candidate < best_S:
                best_S = S_candidate
                best_theta_in = th_in
                best_theta_out = th_out
                # Store the rotated centers for this best candidate
                # We need to re-rotate if we find a better one later, 
                # but for now we can just store th_in and recompute at the end.
                
    # Reconstruct the best solution
    c_in = np.array([[np.cos(best_theta_in), -np.sin(best_theta_in)], [np.sin(best_theta_in), np.cos(best_theta_in)]])
    final_centers = centers @ c_in.T
    
    # Add a small safety margin to S to ensure no floating point violations
    final_S = best_S * 1.001
    
    # Format inner: [[x, y, angle_degrees], ...]
    # Angle is in degrees
    angle_deg = np.rad2deg(best_theta_in)
    inner = [[float(cx), float(cy), float(angle_deg)] for cx, cy in final_centers]
    
    # Outer: center (0,0), side, angle
    outer_angle_deg = np.rad2deg(best_theta_out)
    
    return inner, (0.0, 0.0), float(final_S), float(outer_angle_deg)

# EVOLVE-BLOCK-END
