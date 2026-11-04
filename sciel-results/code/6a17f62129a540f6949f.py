import numpy as np
import time
from scipy.optimize import minimize

N = 11
MIN_DIST = np.sqrt(3)
# Unit hexagon vertices (flat-top, angle 0)
# Vertices at angles 0, 60, 120, 180, 240, 300
HEX_VERTS = np.array([
    [1.0, 0.0],
    [0.5, np.sqrt(3)/2],
    [-0.5, np.sqrt(3)/2],
    [-1.0, 0.0],
    [-0.5, -np.sqrt(3)/2],
    [0.5, -np.sqrt(3)/2]
])

def construct():
    # Normal vectors for the 6 sides of a regular hexagon
    # We will optimize the rotation of the outer hexagon
    
    def cost_func(params):
        # params: [x1, y1, ..., x11, y11, theta_outer]
        centers = params[:22].reshape(-1, 2)
        theta = params[22]
        
        # Center the points at the origin
        centroid = np.mean(centers, axis=0)
        centers_centered = centers - centroid
        
        # Generate all vertices of the inner hexagons
        # Each hexagon has 6 vertices
        # v_{i,m} = c_i + HEX_VERTS[m]
        # We can vectorize this:
        # centers_centered: (11, 2)
        # HEX_VERTS: (6, 2)
        # vertices: (11, 6, 2)
        vertices = centers_centered[:, np.newaxis, :] + HEX_VERTS[np.newaxis, :, :]
        
        # Flatten vertices to (66, 2)
        verts_flat = vertices.reshape(-1, 2)
        
        # Calculate normals for the outer hexagon at rotation theta
        # Normals at angles theta, theta+60, ..., theta+300
        angles = theta + np.linspace(0, 2*np.pi, 7)[:-1]
        normals = np.array([[np.cos(a), np.sin(a)] for a in angles]) # (6, 2)
        
        # Project vertices onto normals
        # verts_flat: (66, 2)
        # normals: (6, 2)
        # projections: (66, 6)
        projections = verts_flat @ normals.T
        
        # We need max |projection| for each normal
        abs_projections = np.abs(projections) # (66, 6)
        max_projections_per_normal = np.max(abs_projections, axis=0) # (6,)
        
        # Use soft-max to smooth the max over the 6 normals
        # L = max_j max_projections_per_normal[j] / (sqrt(3)/2)
        # Soft max: (1/k) * ln(sum(exp(k * x)))
        k = 10.0 # Sharpness
        # To avoid overflow, subtract max
        max_val = np.max(max_projections_per_normal)
        soft_max_val = np.log(np.sum(np.exp(k * (max_projections_per_normal - max_val)))) / k + max_val
        
        L_enc = soft_max_val / (np.sqrt(3) / 2)
        
        # Calculate overlap penalty
        # We want distance >= MIN_DIST
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers_centered[i] - centers_centered[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2
        
        # Combine costs
        # Large penalty for overlaps
        return L_enc + 1000.0 * penalty

    # Random restarts
    best_L = float('inf')
    best_params = None
    
    # Initial guess: Hexagonal lattice
    # Generate points in a hexagonal grid
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
    
    for restart in range(30):
        if restart == 0:
            init_pts_current = init_pts.copy()
            init_pts_current += np.random.normal(0, 0.1, init_pts_current.shape)
        else:
            init_pts_current = np.random.uniform(-4, 4, (N, 2))
            
        initial_params = np.hstack([init_pts_current.flatten(), [0.0]])
        
        # Use L-BFGS-B for optimization
        res = minimize(cost_func, initial_params, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if res.fun < best_L:
            # Extract L_enc from the best params to verify
            final_params = res.x
            centers = final_params[:22].reshape(-1, 2)
            theta = final_params[22]
            
            # Recalculate L_enc exactly (not soft)
            centroid = np.mean(centers, axis=0)
            centers_centered = centers - centroid
            vertices = centers_centered[:, np.newaxis, :] + HEX_VERTS[np.newaxis, :, :]
            verts_flat = vertices.reshape(-1, 2)
            
            angles = theta + np.linspace(0, 2*np.pi, 7)[:-1]
            normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
            
            projections = verts_flat @ normals.T
            max_projections_per_normal = np.max(np.abs(projections), axis=0)
            L_enc = np.max(max_projections_per_normal) / (np.sqrt(3) / 2)
            
            # Check for overlaps
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(centers_centered[i] - centers_centered[j])
                    if d < MIN_DIST - 1e-5:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                best_L = L_enc
                best_params = final_params
                if best_L < 3.5: # Theoretical min is likely around 3-4
                    pass

    # If no valid solution found, fallback
    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Construct the output
    inner_pts = best_params[:22].reshape(-1, 2)
    # The centers are already centered at 0,0 because we subtracted centroid in cost func?
    # Wait, the cost function centered them, but the optimizer works on the raw parameters.
    # So best_params[:22] are the raw coordinates.
    # We should return them centered at 0,0 for the outer hexagon at 0,0.
    centroid = np.mean(inner_pts, axis=0)
    inner_pts_centered = inner_pts - centroid
    
    inner = [[float(x), float(y), 0.0] for x, y in inner_pts_centered]
    
    outer_side = best_L
    outer_angle = best_params[22]
    
    return inner, (0.0, 0.0), outer_side, np.degrees(outer_angle)

