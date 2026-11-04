import numpy as np
import time
from scipy.optimize import minimize, basinhopping

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
        
        # Center the points at the origin to assume the outer hexagon is centered there
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Calculate the side length L of the smallest enclosing regular hexagon (centered at 0,0)
        # L = max( max_i |p_i . n_k| ) / (sqrt(3)/2)
        projections = np.abs(points_centered @ normals.T)
        max_proj = np.max(projections)
        L_enc = max_proj / (np.sqrt(3) / 2)
        
        # Calculate overlap penalty
        # We want distance >= MIN_DIST
        # Use a smooth penalty to help gradient-free optimizers
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points_centered[i] - points_centered[j])
                if d < MIN_DIST:
                    # Quadratic penalty
                    penalty += (MIN_DIST - d) ** 2
        
        # Combine costs
        # Weight penalty heavily to ensure disjointness
        return L_enc + 500.0 * penalty

    # Random restarts to find a good local minimum
    best_L = float('inf')
    best_params = None
    
    # Initial guess: place points in a hexagonal grid pattern roughly
    for restart in range(30):
        # Random initialization
        if restart == 0:
            # Hexagonal lattice initialization
            pts = []
            for q in range(-3, 4):
                for r in range(-3, 4):
                    x = 1.5 * q
                    y = np.sqrt(3)/2 * q + np.sqrt(3) * r
                    pts.append([x, y])
            pts = np.array(pts)
            dists = np.linalg.norm(pts, axis=1)
            indices = np.argsort(dists)[:N]
            init_pts = pts[indices]
            init_pts += np.random.normal(0, 0.1, init_pts.shape)
        else:
            init_pts = np.random.uniform(-3, 3, (N, 2))
            
        initial_params = init_pts.flatten()
        
        # Use Basin Hopping with Nelder-Mead as local minimizer
        try:
            res = basinhopping(cost_func, initial_params, 
                               minimizer_kwargs={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4},
                               stepsize=0.5, 
                               niter=15,
                               disp=False)
            
            final_pts = res.x.reshape(-1, 2)
            centroid = np.mean(final_pts, axis=0)
            final_pts_centered = final_pts - centroid
            projections = np.abs(final_pts_centered @ normals.T)
            max_proj = np.max(projections)
            L_enc = max_proj / (np.sqrt(3) / 2)
            
            # Check for overlaps
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(final_pts_centered[i] - final_pts_centered[j])
                    if d < MIN_DIST - 1e-5:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and L_enc < best_L:
                best_L = L_enc
                best_params = res.x
        except Exception:
            continue

    # If no valid solution found, fallback to a safe large solution
    if best_params is None:
        # Return the seed solution or a large one
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Construct the output
    # The inner hexagons are at best_params positions (centered at 0,0)
    # The outer hexagon side length is best_L + 1
    # The outer hexagon center is (0,0)
    # The inner hexagons have angle 0.0
    
    inner_pts = best_params.reshape(-1, 2)
    inner = [[float(x), float(y), 0.0] for x, y in inner_pts]
    
    outer_side = best_L + 1.0
    
    return inner, (0.0, 0.0), outer_side, 0.0
