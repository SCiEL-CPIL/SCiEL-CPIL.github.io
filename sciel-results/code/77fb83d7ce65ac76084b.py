import numpy as np
import time
from scipy.optimize import minimize

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
        
        # Calculate widths in 6 directions
        # w_k = max(n_k . p) - min(n_k . p)
        projections = points @ normals.T
        max_proj = np.max(projections, axis=0)
        min_proj = np.min(projections, axis=0)
        widths = max_proj - min_proj
        
        # Minimal side length L such that L * sqrt(3) >= width_k + sqrt(3) (inner hex width)
        # L >= (width_k + sqrt(3)) / sqrt(3) = width_k / sqrt(3) + 1
        L_candidates = widths / np.sqrt(3) + 1.0
        L_enc = np.max(L_candidates)
        
        # Calculate overlap penalty
        # We want distance >= MIN_DIST
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2
        
        # Combine costs
        # We want to minimize L_enc, but heavily penalize overlaps
        # If there are overlaps, the solution is invalid, so we add a large penalty
        return L_enc + 100.0 * penalty

    # Random restarts to find a good local minimum
    best_L = float('inf')
    best_params = None
    
    # Initial guess: place points in a hexagonal grid pattern roughly
    # 11 points.
    # Hexagonal lattice initialization is good.
    
    for restart in range(100): # Increase restarts
        # Random initialization
        if restart < 10:
            # Hexagonal lattice initialization
            # Generate points in a hexagonal grid
            pts = []
            for q in range(-3, 4):
                for r in range(-3, 4):
                    # Convert axial to cartesian
                    x = 1.5 * q
                    y = np.sqrt(3)/2 * q + np.sqrt(3) * r
                    pts.append([x, y])
            # Select 11 closest to center
            pts = np.array(pts)
            dists = np.linalg.norm(pts, axis=1)
            indices = np.argsort(dists)[:N]
            init_pts = pts[indices]
            # Add some noise
            init_pts += np.random.normal(0, 0.1, init_pts.shape)
        else:
            # Random initialization in a circle
            r = np.random.uniform(0, 3, N)
            theta = np.random.uniform(0, 2*np.pi, N)
            init_pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
            
        initial_params = init_pts.flatten()
        
        # Use Nelder-Mead for optimization
        res = minimize(cost_func, initial_params, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        # Extract L_enc from the best params
        final_pts = res.x.reshape(-1, 2)
        
        # Check for overlaps
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(final_pts[i] - final_pts[j])
                if d < MIN_DIST - 1e-5:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Recalculate L_enc accurately
            projections = final_pts @ normals.T
            max_proj = np.max(projections, axis=0)
            min_proj = np.min(projections, axis=0)
            widths = max_proj - min_proj
            L_candidates = widths / np.sqrt(3) + 1.0
            L_enc = np.max(L_candidates)
            
            if L_enc < best_L:
                best_L = L_enc
                best_params = res.x

    # If no valid solution found, fallback to a safe large solution
    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Construct the output
    # The inner hexagons are at best_params positions
    # We need to center them?
    # The cost function did NOT center the points. It calculated widths directly.
    # The outer hexagon center can be chosen to minimize the side length.
    # For each direction k, the center coordinate c_k should be the midpoint of the projection range.
    # c . n_k = (max_proj_k + min_proj_k) / 2
    # We have 6 equations for 2 unknowns (c_x, c_y). We can use least squares.
    
    inner_pts = best_params.reshape(-1, 2)
    
    # Calculate the optimal center for the outer hexagon
    projections = inner_pts @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    midpoints = (max_proj + min_proj) / 2.0
    
    # Solve normals @ c = midpoints
    # normals is 6x2, midpoints is 6x1
    # c = pinv(normals) @ midpoints
    center = np.linalg.pinv(normals) @ midpoints
    
    # Recalculate final L_enc with this center?
    # Actually, L_enc derived from widths is independent of the center shift, as long as the center is within the feasible region.
    # The width-based L is the minimal L for *some* center.
    # So best_L is the correct side length.
    
    inner = [[float(x), float(y), 0.0] for x, y in inner_pts]
    
    outer_side = best_L
    
    return inner, (float(center[0]), float(center[1])), outer_side, 0.0
