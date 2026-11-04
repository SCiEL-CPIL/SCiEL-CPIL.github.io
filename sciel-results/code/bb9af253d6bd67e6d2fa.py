import numpy as np
import time

N = 11
MIN_DIST = np.sqrt(3)

def construct():
    # Normal vectors for the 6 sides of a regular hexagon (flat-top orientation)
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    def cost_func(params):
        # params is a 1D array of 22 elements: [x1, y1, x2, y2, ..., x11, y11]
        points = params.reshape(-1, 2)
        
        # Center the points at the origin
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Calculate the side length L of the smallest enclosing regular hexagon (centered at 0,0)
        # Distance from center to side is L * sqrt(3) / 2
        # Projection of points onto normals must be <= L * sqrt(3) / 2
        # L = max( max_i |p_i . n_k| ) / (sqrt(3)/2)
        
        projections = np.abs(points_centered @ normals.T)
        max_proj = np.max(projections)
        L_enc = max_proj / (np.sqrt(3) / 2)
        
        # Calculate overlap penalty
        # We want distance >= MIN_DIST
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points_centered[i] - points_centered[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2
        
        # Combine costs
        # We want to minimize L_enc, but heavily penalize overlaps
        return L_enc + 100.0 * penalty

    def get_L(params):
        points = params.reshape(-1, 2)
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        projections = np.abs(points_centered @ normals.T)
        max_proj = np.max(projections)
        return max_proj / (np.sqrt(3) / 2)

    def is_valid(params):
        points = params.reshape(-1, 2)
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points_centered[i] - points_centered[j])
                if d < MIN_DIST - 1e-5:
                    return False
        return True

    best_L = float('inf')
    best_params = None
    
    # Initial guess: Hexagonal lattice for 11 points
    # 1 center, 6 at dist 1, 4 at dist 2
    # Axial coordinates for hex grid
    # q, r
    # Center: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,1)
    # We need 4 from Ring 2. To maintain symmetry, pick 4 that are 90 deg apart? 
    # Hex grid doesn't have 90 deg symmetry. 
    # Let's just take the first 4 in the list, or spread them out.
    # Actually, a better cluster for 11 is often derived from the 19-point hex (side 3) by removing 8 points.
    # Removing the 8 "corners" of the side-3 hex?
    # Side 3 hex has 19 points.
    # If we remove the 6 points at distance 2 in the "directions" of the sides?
    # Let's just generate the 19 points and keep the 11 closest to center.
    
    pts_all = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Axial to Cartesian (pointy-top or flat-top? Let's use standard axial)
            # x = size * (sqrt(3) * q + sqrt(3)/2 * r)
            # y = size * (3/2 * r)
            # For unit side length, distance between adjacent centers is sqrt(3)?
            # No, distance between centers of adjacent unit hexagons is sqrt(3).
            # So size in axial coords should map to distance sqrt(3).
            # Standard axial to pixel (pointy-top):
            # x = sqrt(3) * (q + r/2)
            # y = 3/2 * r
            x = np.sqrt(3) * (q + r/2.0)
            y = 1.5 * r
            pts_all.append([x, y])
            
    pts_all = np.array(pts_all)
    dists = np.linalg.norm(pts_all, axis=1)
    indices = np.argsort(dists)[:N]
    init_pts = pts_all[indices].copy()
    
    # Add small noise to break symmetry if needed, or just use this
    init_params = init_pts.flatten()

    from scipy.optimize import minimize

    # Run optimization from lattice start
    for restart in range(20):
        if restart == 0:
            current_init = init_params.copy()
        else:
            # Perturb the lattice start
            current_init = init_params + np.random.normal(0, 0.5, 2*N)
            
        # Nelder-Mead is robust to non-differentiable max
        res = minimize(cost_func, current_init, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if res.success:
            final_pts = res.x.reshape(-1, 2)
            if is_valid(res.x):
                L_val = get_L(res.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res.x
                    # Early stop if we get very close to theoretical min?
                    # Theoretical min for 11 hexagons?
                    # 1 hex: L=1.
                    # 7 hex: L=3? (1 center, 6 around).
                    #   Center at 0. Neighbors at dist sqrt(3).
                    #   Max proj of neighbor: sqrt(3) * cos(0) = sqrt(3).
                    #   L = sqrt(3) / (sqrt(3)/2) + 1 = 2 + 1 = 3.
                    #   So 7 hexagons fit in L=3.
                    # 11 hexagons: Add 4 more.
                    #   They will extend further out.
                    #   Likely L > 3.
                    #   Maybe L ~ 3.5 - 4.0?
                    pass

    # If no valid solution found, fallback
    if best_params is None:
        # Use the initial lattice as a valid fallback (it should be valid)
        if is_valid(init_params):
            best_params = init_params
            best_L = get_L(init_params)
        else:
            # Safe large solution
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0

    # Construct the output
    inner_pts = best_params.reshape(-1, 2)
    inner = [[float(x), float(y), 0.0] for x, y in inner_pts]
    
    outer_side = best_L + 1.0

    return inner, (0.0, 0.0), outer_side, 0.0
