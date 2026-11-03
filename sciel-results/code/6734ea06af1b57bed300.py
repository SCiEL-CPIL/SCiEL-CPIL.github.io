import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Helper: get vertices of a regular hexagon with side length 1
    def hex_vertices(cx, cy, angle_rad):
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    # Generate 3x4 grid of hexagon centers
    # This is a compact packing
    centers = []
    sqrt3 = math.sqrt(3)
    for j in range(3):
        for i in range(4):
            x = i * sqrt3 + j * sqrt3 / 2
            y = j * 1.5
            centers.append((x, y))
    centers = np.array(centers)
    
    # Center the grid around the origin
    grid_center = np.mean(centers, axis=0)
    centers -= grid_center

    # Objective function to minimize
    def objective(params):
        alpha, cx, cy, beta = params
        
        # Rotate inner centers
        c_alpha = math.cos(alpha)
        s_alpha = math.sin(alpha)
        rot_matrix = np.array([[c_alpha, -s_alpha], [s_alpha, c_alpha]])
        rotated_centers = centers @ rot_matrix.T
        
        # Get all vertices of inner hexagons
        # Vectorized: create all vertices at once?
        # 12 hexagons * 6 vertices = 72 vertices
        all_verts = []
        for cx_i, cy_i in rotated_centers:
            verts = hex_vertices(cx_i, cy_i, alpha)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # Outer hexagon normals
        n_betas = []
        for k in range(6):
            a = beta + k * math.pi / 3
            n_betas.append((math.cos(a), math.sin(a)))
        n_betas = np.array(n_betas)
        
        # Calculate max projection of inner vertices onto outer normals
        c = np.array([cx, cy])
        rel_verts = all_verts - c
        
        # Projections: (72, 6)
        projections = rel_verts @ n_betas.T
        
        max_proj = np.max(projections)
        
        # S = 2/sqrt(3) * max_proj
        S = (2.0 / sqrt3) * max_proj
        
        return S

    # Initial guess
    # We try several initial angles to escape local minima
    best_score = float('inf')
    best_params = None
    
    # Starting angles for inner cluster rotation and outer hexagon rotation
    # Symmetry of hexagon is 60 degrees (pi/3)
    start_alphas = [0.0, math.pi/6, math.pi/3]
    start_betas = [0.0, math.pi/6, math.pi/3]
    
    for alpha_init in start_alphas:
        for beta_init in start_betas:
            x0 = [alpha_init, 0.0, 0.0, beta_init]
            
            # Use Nelder-Mead for robustness with non-smooth max function
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break

    if best_params is None:
        # Fallback: use the first result or a default
        # Just run one more time to ensure we have something
        res = minimize(objective, [0.0, 0.0, 0.0, 0.0], method='Nelder-Mead', options={'maxiter': 100})
        best_params = res.x
        best_score = res.fun

    alpha, cx, cy, beta = best_params
    
    # Reconstruct inner hexagons
    inner = []
    c_alpha = math.cos(alpha)
    s_alpha = math.sin(alpha)
    rot_matrix = np.array([[c_alpha, -s_alpha], [s_alpha, c_alpha]])
    rotated_centers = centers @ rot_matrix.T
    
    for cx_i, cy_i in rotated_centers:
        # Convert angle to degrees
        angle_deg = math.degrees(alpha)
        inner.append([float(cx_i), float(cy_i), angle_deg])
        
    # Return
    return inner, (float(cx), float(cy)), float(best_score), float(math.degrees(beta))
# EVOLVE-BLOCK-END
