import numpy as np
import math
import itertools
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590
    
    sqrt3 = math.sqrt(3)
    
    # Lattice vectors
    # a = (sqrt(3), 0)
    # b = (sqrt(3)/2, 3/2)
    
    # Generate Ring 0, 1, 2
    centers = []
    # Ring 0
    centers.append((0, 0))
    
    # Ring 1
    ring1 = []
    for i in range(6):
        # Axial coords (q, r)
        # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
        q, r = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)][i]
        x = sqrt3 * (q + r/2)
        y = 1.5 * r
        ring1.append((x, y))
    centers.extend(ring1)
    
    # Ring 2
    ring2 = []
    # Axial coords for Ring 2
    # (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
    coords2 = [(2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)]
    for q, r in coords2:
        x = sqrt3 * (q + r/2)
        y = 1.5 * r
        ring2.append((x, y))
        
    # We need to pick 5 from ring2
    # And rotate the outer hexagon
    
    best_s = float('inf')
    best_inner = None
    best_angle = 0.0
    
    # Try all combinations of 5 from 12 in ring2
    # C(12, 5) = 792
    # For each, try a few rotation angles
    
    # Normals for outer hexagon at angle theta
    # n_k = (cos(theta + 30 + 60k), sin(theta + 30 + 60k))
    
    def compute_s(centers, theta):
        max_proj = 0.0
        for cx, cy in centers:
            for k in range(6):
                a = math.radians(theta + 30 + 60*k)
                nx = math.cos(a)
                ny = math.sin(a)
                proj = cx*nx + cy*ny
                if proj > max_proj:
                    max_proj = proj
        return 1.0 + (2.0/sqrt3) * max_proj

    # Search
    for combo in itertools.combinations(range(12), 5):
        if time.time() > deadline:
            break
        selected = [ring2[i] for i in combo]
        current_centers = centers + selected
        
        # Try rotations
        # The function s(theta) is piecewise linear. Minima occur at vertices of the "projection polygon".
        # We can just sample theta from 0 to 60 in steps of 5 degrees.
        for theta_deg in range(0, 60, 5):
            s = compute_s(current_centers, theta_deg)
            if s < best_s:
                best_s = s
                best_inner = current_centers
                best_angle = theta_deg
                
    # Refine with L-BFGS-B?
    # If best_s is close to the bound, we might not need to.
    
    # Construct return values
    inner_arr = np.zeros((12, 3))
    for i in range(12):
        inner_arr[i][0] = best_inner[i][0]
        inner_arr[i][1] = best_inner[i][1]
        inner_arr[i][2] = 0.0
        
    outer_arr = np.array([0.0, 0.0, float(best_angle)])
    
    return inner_arr, outer_arr, best_s
