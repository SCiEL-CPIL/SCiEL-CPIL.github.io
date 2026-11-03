import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate 11 positions in a hexagonal lattice
    # Spacing d = sqrt(3)
    d = np.sqrt(3)
    
    # 1 center
    pts = [(0.0, 0.0)]
    
    # Ring 1: 6 hexagons
    for i in range(6):
        angle = np.pi * i / 3.0
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        pts.append((x, y))
        
    # Ring 2: 4 hexagons
    # Total 11. We have 7. Need 4 more.
    # Ring 2 positions are at distance 2*d.
    # There are 12 positions in ring 2.
    # Indices 0..11.
    # To keep symmetry, we might pick every 3rd? 0, 3, 6, 9?
    # Or 0, 1, 2, 3?
    # Let's try a few sets of 4 indices from 0..11.
    
    candidate_indices = [
        [0, 3, 6, 9], # Symmetric cross
        [0, 1, 2, 3], # Clustered
        [0, 2, 4, 6], # Alternating
        [1, 3, 5, 7], # Alternating offset
        [0, 4, 8, 12 % 12], # 0, 4, 8, 0? No, 0,4,8,0 is 3 unique.
        [0, 4, 8, 11], # Random-ish
        [0, 2, 5, 8], # Spread
    ]
    
    best_score = float('inf')
    best_config = None
    best_theta = 0.0
    
    # Precompute point array for current set
    # We will iterate through candidate sets
    
    for indices in candidate_indices:
        if time.time() > deadline:
            break
            
        current_pts = list(pts)
        for idx in indices:
            angle = np.pi * (2 * idx) / 12.0 # 12 positions in ring 2, step 30 deg
            # Wait, ring 2 in hex lattice:
            # The directions are same as ring 1 (6 dirs) plus 6 in between?
            # Yes, the lattice vectors are $a_1, a_2$.
            # Ring 2 contains $2 \times$ the vectors of ring 1 and combinations.
            # Specifically, positions are $2 a_1, 2 a_2, a_1+a_2$, etc.
            # In polar coords, they are at angles $0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330$.
            # Distance is $2d$ for $0, 60, \dots$ and $d\sqrt{3}$? No.
            # Distance to $(2,0)$ in lattice coords is $2d$.
            # Distance to $(1,1)$ in lattice coords (which is angle 30) is $d\sqrt{3}$?
            # Let's check. $a_1 = (d, 0)$. $a_2 = (d/2, d\sqrt{3}/2)$.
            # $a_1 + a_2 = (1.5d, d\sqrt{3}/2)$.
            # Magnitude: $\sqrt{2.25 d^2 + 0.75 d^2} = \sqrt{3 d^2} = d\sqrt{3}$.
            # So the "corner" positions of the hexagonal cluster (1+6) are at distance $d\sqrt{3}$?
            # No, the 1+6 cluster occupies $a_1, a_2, a_1+a_2$, etc.
            # The points in Ring 1 are at distance $d$.
            # The points in Ring 2 are at distance $2d$ (along axes) and $d\sqrt{3}$ (diagonals)?
            # Actually, the 1+6 cluster is a hexagon of side 2 (in hex units).
            # The "corners" of this cluster are the hexagons at $a_1+a_2$, etc.
            # Their distance from center is $d\sqrt{3}$.
            # The "side midpoints" of the cluster are at $2a_1$? No, $2a_1$ is further out.
            # Let's just list the 12 positions of the second shell of a hex lattice.
            # Vectors:
            # (2, 0), (1, 1), (0, 2), (-1, 1), (-2, 0), (-1, -1), (0, -2), (1, -1), (2, 0)... wait.
            # The 12 vectors are:
            # (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1) -> these are 8?
            # No, the second ring has 12 hexagons.
            # Coordinates in basis $a_1, a_2$:
            # (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1) is only 8.
            # Where are the other 4?
            # (1,-1)? No.
            # The hexagon of side 2 has perimeter 12.
            # Vertices are at distance $2d$?
            # Let's just use the polar angles.
            # The 12 positions are at angles $0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330$.
            # Distances:
            # Angle 0: $2d$.
            # Angle 30: $d\sqrt{3}$.
            # Angle 60: $2d$.
            # Angle 90: $d\sqrt{3}$.
            # And so on.
            
            dist = 2*d if idx % 2 == 0 else d * np.sqrt(3)
            x = dist * np.cos(angle)
            y = dist * np.sin(angle)
            current_pts.append((x, y))
            
        # Optimize rotation theta
        # S(theta) = max_p max_k (p . n_k) / (sqrt(3)/2)
        # n_k are normals at theta + k*60 deg.
        
        # We can sample theta from 0 to 60 deg.
        # Since the problem is symmetric, 0-30 might be enough?
        # Let's sample 0 to 60 in steps of 1 deg.
        
        pts_arr = np.array(current_pts)
        
        best_theta_local = 0.0
        min_side_local = float('inf')
        
        for theta_deg in range(0, 61, 1):
            if time.time() > deadline:
                break
            theta = np.radians(theta_deg)
            # Normals
            angles = theta + np.arange(6) * np.pi / 3.0
            normals = np.column_stack((np.cos(angles), np.sin(angles)))
            
            # Project points onto normals
            # pts_arr shape (N, 2), normals shape (6, 2)
            # proj shape (N, 6)
            proj = pts_arr @ normals.T
            
            # Max projection for each normal
            max_proj = np.max(proj, axis=0)
            
            # The required circumradius R is max(max_proj) / cos(30)
            # cos(30) = sqrt(3)/2
            R = np.max(max_proj) / (np.sqrt(3) / 2.0)
            
            if R < min_side_local:
                min_side_local = R
                best_theta_local = theta_deg
                
        if min_side_local < best_score:
            best_score = min_side_local
            best_config = [p + (0.0,) for p in current_pts] # (x, y, angle)
            best_theta = best_theta_local
            
    # Final Polish: Refine theta around best_theta_local
    if best_config is not None:
        # Fine search
        best_theta_fine = best_theta
        min_side_fine = best_score
        pts_arr = np.array([p[:2] for p in best_config])
        
        for theta_deg_10 in range(int(best_theta * 10) - 10, int(best_theta * 10) + 11):
            theta_deg = theta_deg_10 / 10.0
            theta = np.radians(theta_deg)
            angles = theta + np.arange(6) * np.pi / 3.0
            normals = np.column_stack((np.cos(angles), np.sin(angles)))
            proj = pts_arr @ normals.T
            max_proj = np.max(proj, axis=0)
            R = np.max(max_proj) / (np.sqrt(3) / 2.0)
            if R < min_side_fine:
                min_side_fine = R
                best_theta_fine = theta_deg
                
        best_score = min_side_fine
        best_theta = best_theta_fine

    # Return best config
    # inner: list of [x, y, angle_degrees]
    # center: (0,0)
    # side: best_score
    # angle: best_theta
    
    inner = [[x, y, 0.0] for x, y, _ in best_config]
    return inner, (0.0, 0.0), best_score, best_theta

# EVOLVE-BLOCK-END
