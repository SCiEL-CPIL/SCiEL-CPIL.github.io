import numpy as np

def hexagon_packing_11():
    # Define the 11 centers in local frame (Honeycomb)
    # Center
    centers = [[0.0, 0.0]]
    
    # Ring 1 (6 hexes)
    for i in range(6):
        angle = np.deg2rad(i * 60)
        centers.append([2.0 * np.cos(angle), 2.0 * np.sin(angle)])
        
    # Ring 2 (4 hexes)
    # We try two symmetric configurations
    # Config 1: Angles 30, 150, 210, 330 (indices 0, 2, 3, 5 of the 6 ring-2 spots)
    # Ring 2 spots are at angles 30, 90, 150, 210, 270, 330
    r2_angles = [30, 90, 150, 210, 270, 330]
    r2_dist = 2.0 * np.sqrt(3)
    
    configs = [
        [30, 150, 210, 330], # Symmetric across x and y
        [90, 150, 210, 270]  # Symmetric across y, not x?
    ]
    
    best_S = float('inf')
    best_theta = 0
    best_centers = None
    best_config_idx = 0
    
    # Outer hex normals (angles in degrees)
    # Flat-top hexagon normals are at 30, 90, 150, 210, 270, 330
    n_angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.array([[np.cos(a), np.sin(a)] for a in n_angles])
    
    for c_idx, config_angles in enumerate(configs):
        # Add ring 2 centers
        c_local = centers.copy()
        for a_deg in config_angles:
            a_rad = np.deg2rad(a_deg)
            c_local.append([r2_dist * np.cos(a_rad), r2_dist * np.sin(a_rad)])
        
        c_local = np.array(c_local)
        
        # Scan theta
        # Symmetry period is 60 degrees. We can scan 0 to 60.
        # Due to symmetry of the hexagon container, 0 to 30 might be enough if we handle signs?
        # The function A_req(theta) is periodic with 60.
        # Let's scan 0 to 60 with step 0.1 deg.
        for theta_deg in np.arange(0, 60, 0.1):
            theta_rad = np.deg2rad(theta_deg)
            c, s = np.cos(theta_rad), np.sin(theta_rad)
            R = np.array([[c, -s], [s, c]])
            
            # Rotate centers
            c_rot = (R @ c_local.T).T
            
            # Calculate max projection on all normals
            # c_rot shape (11, 2), normals shape (6, 2)
            # Projections: (11, 6)
            projections = c_rot @ normals.T
            max_proj = np.max(projections)
            
            # S = 1 + (2/sqrt(3)) * max_proj
            S = 1.0 + (2.0 / np.sqrt(3.0)) * max_proj
            
            if S < best_S:
                best_S = S
                best_theta = theta_rad
                best_centers = c_rot
                best_config_idx = c_idx

    # Construct return values
    # Inner hexagons: (x, y, angle)
    # Angle is 0 for all
    inner_data = np.zeros((11, 3))
    inner_data[:, 0] = best_centers[:, 0]
    inner_data[:, 1] = best_centers[:, 1]
    inner_data[:, 2] = 0.0
    
    # Outer hexagon: (x, y, angle)
    outer_data = np.array([0.0, 0.0, 0.0])
    
    return inner_data, outer_data, best_S
