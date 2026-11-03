import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0

    # Constants
    d = math.sqrt(3)  # Distance between centers of adjacent hexagons
    # Unit hexagon vertices (pointy-top, centered at origin)
    # Angles: 0, 60, 120, 180, 240, 300
    v_base = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])

    def get_vertices(center, angle):
        """Get vertices of a unit hexagon with given center and rotation angle."""
        c = np.array(center)
        theta = angle
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        # Rotate base vertices
        verts = v_base.T @ R
        # Translate
        return verts + c[:, np.newaxis]

    def objective(params):
        """
        params: [x1, y1, x2, y2, ..., x11, y11, theta]
        """
        if time.time() > deadline - safety_margin:
            return float('inf')
        
        centers = params[:22].reshape(11, 2)
        theta = params[22]
        
        # 1. Compute h_max (max projection on 6 normals)
        # Normals for outer hexagon
        k_angles = np.arange(6) * math.pi / 3
        angles = theta + k_angles
        normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1) # (6, 2)
        
        # Get all vertices of all 11 hexagons
        # All hexagons are aligned with theta
        all_verts = np.zeros((11, 6, 2))
        for i in range(11):
            all_verts[i] = get_vertices(centers[i], theta)
        
        all_verts_flat = all_verts.reshape(-1, 2) # (66, 2)
        
        # Projections: (6, 66)
        dots = np.sum(normals[:, np.newaxis, :] * all_verts_flat[np.newaxis, :, :], axis=-1)
        abs_dots = np.abs(dots)
        max_proj = np.max(abs_dots)
        
        # Side length S = 2/sqrt(3) * max_proj
        # We minimize max_proj, which is proportional to S
        cost = max_proj
        
        # 2. Penalty for overlapping hexagons
        # Min distance between centers is sqrt(3)
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < d:
                    penalty += (d - dist)**2
        
        # Add penalty
        # If penalty is large, the configuration is invalid
        # We use a large multiplier to push towards valid configs
        return cost + 1000.0 * penalty

    # Initial guess: Hexagonal lattice
    # Center + Ring 1 (6) + Ring 2 (4 selected)
    # We use the same lattice as the seed program, but we will optimize it.
    
    # Lattice centers
    centers_init = []
    centers_init.append((0.0, 0.0))
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers_init.append((x, y))
    
    # Ring 2
    ring2_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_centers.append((x, y))
    
    # Select 4 from ring 2 to make 11 total
    # The seed program found a specific combination. Let's try a few symmetric ones.
    # Symmetric choice: (0,2), (1,1), (-1,1), (-2,0) ... wait, need to map q,r to index.
    # Let's just pick the first 4 or try a few random ones.
    # Actually, the optimal packing for 11 hexagons is likely the "flower" of 7 plus 4 more.
    # The 4 more should be placed symmetrically.
    
    # Let's try the combination from the seed program's best result if we can infer it, 
    # or just try a few symmetric patterns.
    # Pattern 1: 4 hexagons from ring 2 at angles 0, 90, 180, 270? No, hexagonal symmetry.
    # Ring 2 has 12 hexagons. We need to pick 4.
    # Symmetric pick: indices 0, 3, 6, 9 (every 3rd one).
    # Let's check the angles of ring 2 centers.
    # (2,0) -> angle 0
    # (1,1) -> angle 60? No.
    # Let's just use the first 4 for now, and rely on optimization.
    # Or better, try 3 different symmetric initializations.
    
    inits = []
    # Init 1: 0, 3, 6, 9
    inits.append(ring2_centers[0:4]) # Just first 4, not necessarily symmetric, but let's see
    # Let's construct a symmetric set manually.
    # Ring 2 centers:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (1,-1), (0,-1) ... wait, q,r mapping.
    # Let's just use the list generated above.
    # Let's try picking indices that are symmetric.
    # The 12 centers are at angles: 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330?
    # No, hexagonal lattice ring 2 is not uniform in angle like that.
    # Let's just try 3 random symmetric-looking inits.
    
    # Init 1: The one from seed (we don't know exactly which, but let's try 0,1,4,5)
    # Init 2: 0, 3, 6, 9
    # Init 3: 0, 4, 8, 11
    
    init_sets = [
        [0, 1, 4, 5],
        [0, 3, 6, 9],
        [0, 4, 8, 11]
    ]
    
    best_params = None
    best_cost = float('inf')
    
    for init_set in init_sets:
        centers = list(centers_init[:7])
        for idx in init_set:
            centers.append(ring2_centers[idx])
        
        # Flatten centers
        params_init = np.array(centers).flatten()
        params_init = np.append(params_init, 0.0) # Initial theta = 0
        
        # Run optimization
        try:
            res = minimize(objective, params_init, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
            
            if res.fun < best_cost:
                best_cost = res.fun
                best_params = res.x
        except Exception as e:
            pass
        
        if time.time() > deadline - safety_margin:
            break
    
    # If optimization failed or returned inf, fallback to seed
    if best_params is None or best_cost == float('inf'):
        # Use seed logic
        centers = list(centers_init[:7]) + [ring2_centers[i] for i in [0,1,4,5]]
        inner = [[cx, cy, 0.0] for cx, cy in centers]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Extract best results
    centers_final = best_params[:22].reshape(11, 2)
    theta_final = best_params[22]
    
    # Calculate final side length
    # We need to return the side length S
    # S = 2/sqrt(3) * max_proj
    # Let's recompute max_proj for the final params to get S
    k_angles = np.arange(6) * math.pi / 3
    angles = theta_final + k_angles
    normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    
    all_verts = np.zeros((11, 6, 2))
    for i in range(11):
        all_verts[i] = get_vertices(centers_final[i], theta_final)
    all_verts_flat = all_verts.reshape(-1, 2)
    
    dots = np.sum(normals[:, np.newaxis, :] * all_verts_flat[np.newaxis, :, :], axis=-1)
    abs_dots = np.abs(dots)
    max_proj = np.max(abs_dots)
    side_length = (2.0 / math.sqrt(3)) * max_proj
    
    # Format output
    inner_rows = []
    for i in range(11):
        cx, cy = centers_final[i]
        inner_rows.append([cx, cy, math.degrees(theta_final)])
    
    center = (0.0, 0.0)
    angle_deg = math.degrees(theta_final)
    
    return inner_rows, center, side_length, angle_deg

