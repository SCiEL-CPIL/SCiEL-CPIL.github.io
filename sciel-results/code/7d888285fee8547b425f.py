import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0

def construct():
    # Check if we are already close to the time limit
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate 11 points on a triangular lattice
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    # Scale factor: 1.0 ensures touching, we start slightly expanded to be safe
    # But we want to pack tightly, so we start at 1.0 and let optimizer adjust if needed
    # Actually, starting at 1.001 is safer to avoid initial penalty
    SCALE = 1.001
    
    pts = np.array([
        (math.sqrt(3) * (q + r / 2.0) * SCALE, 1.5 * r * SCALE)
        for q, r in axial
    ])
    
    # Initial best solution: the lattice one
    def calc_S_for_config(pts, beta):
        centroid = np.mean(pts, axis=0)
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        max_S = 0.0
        for nx, ny in normals:
            rel_pts = pts - centroid
            projections = rel_pts[:, 0] * nx + rel_pts[:, 1] * ny
            abs_proj = np.max(np.abs(projections))
            required_S = (abs_proj + APOTHEM) / APOTHEM
            if required_S > max_S:
                max_S = required_S
        return max_S

    best_S = calc_S_for_config(pts, 0.0)
    best_pts = pts.copy()
    best_beta = 0.0
    
    # Define objective function for optimization
    # x: [x1, y1, ..., x11, y11, beta]
    def objective(x):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        pts_opt = x[:2*N].reshape(N, 2)
        beta = x[2*N]
        
        # Calculate S
        S = calc_S_for_config(pts_opt, beta)
        
        # Calculate overlap penalty
        # Min distance between any two centers must be >= sqrt(3)
        min_dist = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts_opt[i] - pts_opt[j])
                if d < min_dist:
                    min_dist = d
        
        if min_dist < math.sqrt(3):
            penalty = 100.0 * (math.sqrt(3) - min_dist) ** 2
        else:
            penalty = 0.0
            
        return S + penalty

    # Try a few initial rotations to help the optimizer
    initial_betas = [0.0, math.pi/12, math.pi/6]
    
    for init_beta in initial_betas:
        x0 = np.concatenate([pts.flatten(), [init_beta]])
        
        try:
            # Use Nelder-Mead for non-smooth objective
            # maxiter limited to prevent timeout
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < best_S and res.fun < float('inf'):
                # Check if the result is valid (no overlap)
                pts_res = res.x[:2*N].reshape(N, 2)
                beta_res = res.x[2*N]
                
                min_dist = float('inf')
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.linalg.norm(pts_res[i] - pts_res[j])
                        if d < min_dist:
                            min_dist = d
                
                if min_dist >= math.sqrt(3) - 1e-5: # Small tolerance
                    S_res = calc_S_for_config(pts_res, beta_res)
                    if S_res < best_S:
                        best_S = S_res
                        best_pts = pts_res
                        best_beta = beta_res
                        
        except Exception:
            pass
            
    # Prepare output
    inner = []
    angle_deg = math.degrees(best_beta)
    for x, y in best_pts:
        inner.append([float(x), float(y), angle_deg])
        
    centroid = np.mean(best_pts, axis=0)
    center = [float(centroid[0]), float(centroid[1])]
    side = float(best_S)
    rotation = angle_deg
    
    return inner, center, side, rotation
