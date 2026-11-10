import numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations

def hexagon_packing_11():
    # Define the 11 centers in local frame (Honeycomb)
    # Center
    centers = [[0.0, 0.0]]
    
    # Ring 1 (6 hexes)
    for i in range(6):
        angle = np.deg2rad(i * 60)
        centers.append([2.0 * np.cos(angle), 2.0 * np.sin(angle)])
        
    # Ring 2 (4 hexes)
    # Ring 2 spots are at angles 30, 90, 150, 210, 270, 330
    r2_angles = [30, 90, 150, 210, 270, 330]
    r2_dist = 2.0 * np.sqrt(3)
    
    # Generate all 15 combinations of 4 spots from the 6 available
    # We can reduce this by symmetry, but 15 is small.
    # Symmetry group of the hexagon is D6.
    # We'll just iterate all and keep the best.
    
    best_S = float('inf')
    best_theta = 0.0
    best_centers = None
    
    # Outer hex normals (angles in degrees)
    # Flat-top hexagon normals are at 30, 90, 150, 210, 270, 330
    n_angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.array([[np.cos(a), np.sin(a)] for a in n_angles])
    
    # Base centers array (first 7)
    base_centers = np.array(centers)
    
    # Pre-compute ring 2 candidate positions
    r2_positions = []
    for a_deg in r2_angles:
        a_rad = np.deg2rad(a_deg)
        r2_positions.append([r2_dist * np.cos(a_rad), r2_dist * np.sin(a_rad)])
    r2_positions = np.array(r2_positions)
    
    # Function to calculate S for a given set of centers and rotation theta
    def calc_S(c_local, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        c_rot = (R @ c_local.T).T
        projections = c_rot @ normals.T
        max_proj = np.max(projections)
        # S = 1 + (2/sqrt(3)) * max_proj
        return 1.0 + (2.0 / np.sqrt(3.0)) * max_proj, c_rot

    # Iterate over all 15 subsets of 4 from 6
    for subset_indices in combinations(range(6), 4):
        # Build local centers
        c_local = np.vstack([base_centers, r2_positions[list(subset_indices)]])
        
        # Optimize theta
        # The function is periodic with 60 deg (pi/3).
        # We can search in [0, pi/3].
        # Use minimize_scalar
        res = minimize_scalar(lambda th: calc_S(c_local, th)[0], bounds=(0, np.pi/3), method='bounded')
        
        if res.success:
            theta_opt = res.x
            S_val, c_rot_opt = calc_S(c_local, theta_opt)
            
            if S_val < best_S:
                best_S = S_val
                best_theta = theta_opt
                best_centers = c_rot_opt

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
