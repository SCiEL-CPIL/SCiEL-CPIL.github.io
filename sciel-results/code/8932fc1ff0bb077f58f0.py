import numpy as np
from itertools import combinations
import time

def hexagon_packing_12():
    start_time = time.time()
    
    # Define hex lattice centers
    # Axial coordinates (q, r)
    # Center
    centers_axial = [(0, 0)]
    
    # Ring 1
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    centers_axial.extend(ring1)
    
    # Ring 2
    ring2 = [
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2), (-2, 1),
        (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -2), (2, -1)
    ]
    
    # We need to choose 5 from ring2
    # To save time, we can limit the search or use symmetry.
    # Due to 6-fold symmetry of the lattice, we can fix one hexagon's position relative to the ring2 selection?
    # Actually, the container is symmetric. The cluster symmetry is broken by the choice of 5.
    # We will iterate over all combinations of 5 from ring2.
    # There are 792 combinations.
    # For each, we optimize theta.
    
    best_s = float('inf')
    best_theta = 0
    best_centers = None
    best_combination = None
    
    # Pre-calculate ring2 axial to cartesian
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])
    
    base_centers = np.array([axial_to_cartesian(q, r) for q, r in centers_axial[:7]])
    ring2_centers = np.array([axial_to_cartesian(q, r) for q, r in ring2])
    
    # Normals of the container (fixed orientation)
    # Vertices at 0, 60, ...
    # Normals at 30, 90, 150
    n0 = np.array([np.cos(np.pi/6), np.sin(np.pi/6)])
    n1 = np.array([0, 1])
    n2 = np.array([-np.cos(np.pi/6), np.sin(np.pi/6)])
    normals = np.array([n0, n1, n2])
    
    # Function to calculate vertices of 12 hexagons given centers and angle
    def get_vertices(centers, angle):
        # centers: (12, 2)
        # angle: scalar
        # Returns (72, 2)
        # Hex vertices relative to center
        v_rel = np.array([
            [np.cos(angle + i * np.pi / 3), np.sin(angle + i * np.pi / 3)] 
            for i in range(6)
        ])
        # (6, 2)
        
        # Broadcast centers to (12, 1, 2)
        c = centers[:, np.newaxis, :]
        # v_rel to (1, 6, 2)
        v = v_rel[np.newaxis, :, :]
        
        # (12, 6, 2)
        verts = c + v
        return verts.reshape(-1, 2)
    
    # Function to calculate minimal s for a given theta
    def calc_s(theta, centers):
        # Rotate centers
        c, s_rot = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[c, -s_rot], [s_rot, c]])
        rotated_centers = centers @ rot_mat.T
        
        # Get vertices
        verts = get_vertices(rotated_centers, theta)
        
        # Project onto normals
        # verts: (72, 2)
        # normals: (3, 2)
        # projections: (72, 3)
        proj = verts @ normals.T
        
        # Widths
        widths = np.max(proj, axis=0) - np.min(proj, axis=0)
        
        # s = max(widths) / sqrt(3)
        return np.max(widths) / np.sqrt(3)
    
    # Iterate over combinations of 5 from ring2
    # To speed up, we can subsample or use a smarter selection.
    # But 792 * (opt steps) might be okay if opt is fast.
    # Let's do a coarse grid for theta first, then refine.
    
    all_combinations = list(combinations(range(12), 5))
    
    for i, comb in enumerate(all_combinations):
        if time.time() - start_time > 550:
            break
            
        selected_ring2 = ring2_centers[list(comb)]
        all_centers = np.vstack((base_centers, selected_ring2))
        
        # Coarse grid for theta
        thetas = np.linspace(0, np.pi/3, 61)
        best_local_s = float('inf')
        best_local_theta = 0
        
        for t in thetas:
            s_val = calc_s(t, all_centers)
            if s_val < best_local_s:
                best_local_s = s_val
                best_local_theta = t
        
        # Refine with Nelder-Mead or finer grid
        # Use a finer grid around best_local_theta
        if best_local_s < best_s:
             # Fine grid
             thetas_fine = np.linspace(best_local_theta - np.pi/3/60, best_local_theta + np.pi/3/60, 201)
             for t in thetas_fine:
                 s_val = calc_s(t, all_centers)
                 if s_val < best_local_s:
                     best_local_s = s_val
                     best_local_theta = t
             
             if best_local_s < best_s:
                 best_s = best_local_s
                 best_theta = best_local_theta
                 best_centers = all_centers
                 best_combination = comb

    if best_centers is None:
        # Fallback
        best_centers = np.vstack((base_centers, ring2_centers[:5]))
        best_theta = 0
        best_s = calc_s(best_theta, best_centers)
        
    # Final rotation
    c, s_rot = np.cos(best_theta), np.sin(best_theta)
    rot_mat = np.array([[c, -s_rot], [s_rot, c]])
    final_centers = best_centers @ rot_mat.T
    
    # Inner hexagons data
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = final_centers[:, 0]
    inner_hex_data[:, 1] = final_centers[:, 1]
    inner_hex_data[:, 2] = np.degrees(best_theta)
    
    # Outer hexagon data
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, best_s

