import numpy as np
import time

N = 11
MIN_DIST = np.sqrt(3)

def construct():
    # Normal vectors for the 6 sides of a regular hexagon (flat-top orientation)
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    base_angles = np.linspace(0, 2*np.pi, 7)[:-1]
    
    def cost_func(params):
        # params is a 1D array of 23 elements: [x1, y1, ..., x11, y11, angle]
        points = params[:2*N].reshape(-1, 2)
        rot = params[2*N]
        
        # Center the points at the origin
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Rotate normals by rot
        angles = base_angles + rot
        normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
        
        # Calculate the side length L of the smallest enclosing regular hexagon
        # L = (max_proj + 1) * 2 / sqrt(3)
        projections = np.abs(points_centered @ normals.T)
        max_proj = np.max(projections)
        L_enc = (max_proj + 1.0) * 2.0 / np.sqrt(3)
        
        # Calculate overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points_centered[i] - points_centered[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2
        
        return L_enc + 1000.0 * penalty

    best_L = float('inf')
    best_params = None
    
    from scipy.optimize import minimize
    
    # Perform multiple random restarts
    for restart in range(100):
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
            init_rot = 0.0
        else:
            init_pts = np.random.uniform(-3, 3, (N, 2))
            init_rot = np.random.uniform(0, 2*np.pi)
            
        initial_params = np.concatenate([init_pts.flatten(), [init_rot]])
        
        # Use SLSQP for better convergence
        res = minimize(cost_func, initial_params, method='SLSQP', 
                       options={'maxiter': 1000, 'ftol': 1e-6})
        
        # Also try Nelder-Mead as fallback or alternative
        res_nm = minimize(cost_func, initial_params, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        # Pick the better result
        if res.fun < res_nm.fun:
            best_res = res
        else:
            best_res = res_nm
            
        if best_res.success:
            final_params = best_res.x
            final_pts = final_params[:2*N].reshape(-1, 2)
            final_rot = final_params[2*N]
            
            centroid = np.mean(final_pts, axis=0)
            final_pts_centered = final_pts - centroid
            
            angles = base_angles + final_rot
            normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
            projections = np.abs(final_pts_centered @ normals.T)
            max_proj = np.max(projections)
            L_enc = (max_proj + 1.0) * 2.0 / np.sqrt(3)
            
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
                best_params = final_params

    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner_pts = best_params[:2*N].reshape(-1, 2)
    # Shift to center at (0,0) for output? The task says outer center (x,y).
    # We assumed outer center is (0,0) in cost function.
    # So inner points are relative to (0,0).
    inner = [[float(x), float(y), 0.0] for x, y in inner_pts]
    
    outer_side = best_L
    outer_rot = float(best_params[2*N])
    
    return inner, (0.0, 0.0), outer_side, outer_rot
