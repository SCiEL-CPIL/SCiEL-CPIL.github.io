import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    # Unit hexagon vertices (circumradius = 1)
    hex_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    hex_vertices = np.column_stack((np.cos(hex_angles), np.sin(hex_angles)))
    
    # Normal directions for the outer hexagon (for support function)
    outer_normals_angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    outer_normals = np.column_stack((np.cos(outer_normals_angles), np.sin(outer_normals_angles)))
    
    def get_hex_vertices(center, angle_deg):
        """Get 6 vertices of a unit hexagon at center with rotation angle_deg"""
        angle = np.deg2rad(angle_deg)
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])
        return center + hex_vertices @ rot_matrix.T
    
    def objective(params):
        r1, r2, theta = params
        theta_rad = np.deg2rad(theta)
        
        # Build centers array
        centers = np.zeros((N, 2))
        # Ring 0: center (index 0)
        # Ring 1: 5 hexagons at radius r1 (indices 1-5)
        for k in range(5):
            angle = theta_rad + 2 * np.pi * k / 5
            centers[k+1] = [r1 * np.cos(angle), r1 * np.sin(angle)]
        # Ring 2: 6 hexagons at radius r2 (indices 6-11)
        for k in range(6):
            angle = theta_rad + 2 * np.pi * k / 6
            centers[k+6] = [r2 * np.cos(angle), r2 * np.sin(angle)]
        
        # Check pairwise non-overlap using support function
        # Two convex polygons don't overlap iff there exists a separating axis
        # For hexagons, check all 12 normal directions (6 from each hexagon's edges)
        viol_sum = 0.0
        for i in range(N):
            for j in range(i+1, N):
                # Get edge normals for hexagon i and j
                ci = centers[i]
                cj = centers[j]
                diff = ci - cj
                
                # Hexagon i's edge normals
                normals_i = np.column_stack((
                    np.cos(hex_angles + np.pi/2),
                    np.sin(hex_angles + np.pi/2)
                ))
                # Hexagon j's edge normals (same orientation)
                normals_j = normals_i
                
                # Project diff onto all normals
                dots_i = diff @ normals_i.T
                dots_j = diff @ normals_j.T
                
                # The minimum projection tells us the gap
                min_proj = min(np.min(dots_i), np.min(dots_j))
                # For non-overlap, we need the hexagons to be separated
                # Actually, we need to check if the Minkowski difference contains origin
                # Simpler: compute min distance between hexagon i and j
                # Use: two hexagons overlap iff for all separating axes, projections overlap
                
                # Check all 12 axes
                all_normals = np.vstack([normals_i, normals_j])
                # Vertices of i and j
                verts_i = ci + hex_vertices
                verts_j = cj + hex_vertices
                
                overlap = True
                for n in all_normals:
                    proj_i = verts_i @ n
                    proj_j = verts_j @ n
                    # Projections overlap if max(proj_i) >= min(proj_j) and max(proj_j) >= min(proj_i)
                    if np.max(proj_i) < np.min(proj_j) - 1e-9 or np.max(proj_j) < np.min(proj_i) - 1e-9:
                        overlap = False
                        break
                
                if overlap:
                    # Estimate violation magnitude
                    viol_sum += 0.1
    
        # Compute support function for outer hexagon
        max_support = 0.0
        for n in outer_normals:
            for i in range(N):
                verts = get_hex_vertices(centers[i], 0)
                dots = verts @ n
                max_support = max(max_support, np.max(dots))
        
        L = (2.0 / np.sqrt(3.0)) * max_support
        return L + 10.0 * viol_sum
    
    best_L = float('inf')
    best_params = None
    
    for seed in range(30):
        np.random.seed(seed)
        # Reasonable initial guesses
        r1_init = np.random.uniform(1.5, 2.5)
        r2_init = np.random.uniform(2.5, 4.0)
        theta_init = np.random.uniform(0, 360)
        x0 = np.array([r1_init, r2_init, theta_init])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if result.fun < best_L:
                # Verify validity
                r1, r2, theta = result.x
                theta_rad = np.deg2rad(theta)
                centers = np.zeros((N, 2))
                for k in range(5):
                    angle = theta_rad + 2 * np.pi * k / 5
                    centers[k+1] = [r1 * np.cos(angle), r1 * np.sin(angle)]
                for k in range(6):
                    angle = theta_rad + 2 * np.pi * k / 6
                    centers[k+6] = [r2 * np.cos(angle), r2 * np.sin(angle)]
                
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        ci = centers[i]
                        cj = centers[j]
                        verts_i = ci + hex_vertices
                        verts_j = cj + hex_vertices
                        all_normals = np.column_stack((
                            np.cos(hex_angles + np.pi/2),
                            np.sin(hex_angles + np.pi/2)
                        ))
                        for n in all_normals:
                            proj_i = verts_i @ n
                            proj_j = verts_j @ n
                            if np.max(proj_i) >= np.min(proj_j) - 1e-6 and np.max(proj_j) >= np.min(proj_i) - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_L = result.fun
                    best_params = result.x
        except Exception:
            continue
    
    if best_params is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    r1, r2, theta = best_params
    theta_rad = np.deg2rad(theta)
    centers = np.zeros((N, 2))
    for k in range(5):
        angle = theta_rad + 2 * np.pi * k / 5
        centers[k+1] = [r1 * np.cos(angle), r1 * np.sin(angle)]
    for k in range(6):
        angle = theta_rad + 2 * np.pi * k / 6
        centers[k+6] = [r2 * np.cos(angle), r2 * np.sin(angle)]
    
    # Recompute L accurately
    max_support = 0.0
    for n in outer_normals:
        for i in range(N):
            verts = get_hex_vertices(centers[i], 0)
            dots = verts @ n
            max_support = max(max_support, np.max(dots))
    
    L_final = (2.0 / np.sqrt(3.0)) * max_support
    
    inner = [[centers[i, 0], centers[i, 1], 0.0] for i in range(N)]
    return inner, (0.0, 0.0), L_final, 0.0
