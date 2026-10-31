import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0

def construct():
    # Generate 11 points on a triangular lattice
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    SCALE = 1.001
    
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0) * SCALE
        y = 1.5 * r * SCALE
        pts.append((x, y))
        
    pts = np.array(pts)
    centroid = np.mean(pts, axis=0)
    
    # Flatten positions and beta into a single vector
    # x0 = [p0_x, p0_y, ..., p10_x, p10_y, beta]
    x0 = np.zeros(2 * N + 1)
    for i in range(N):
        x0[2*i] = pts[i, 0]
        x0[2*i+1] = pts[i, 1]
    x0[-1] = 0.0 # initial beta
    
    def objective(x):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10
        
        # Extract positions and beta
        p = x[:-1].reshape(N, 2)
        beta = x[-1]
        
        # Calculate centroid of current positions
        c = np.mean(p, axis=0)
        
        # Calculate required S
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
            
        max_S = 0.0
        for nx, ny in normals:
            rel_pts = p - c
            projections = rel_pts[:, 0] * nx + rel_pts[:, 1] * ny
            abs_proj = np.max(np.abs(projections))
            required_S = (abs_proj + APOTHEM) / APOTHEM
            if required_S > max_S:
                max_S = required_S
        
        # Calculate overlap penalty
        # For parallel hexagons, min distance is sqrt(3)
        min_dist = math.sqrt(3)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(p[i] - p[j])
                if d < min_dist:
                    penalty += (min_dist - d) ** 2
        
        return max_S + 10.0 * penalty

    # Optimize
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        best_x = res.x
    except Exception:
        best_x = x0
        
    # Extract best result
    best_pts = best_x[:-1].reshape(N, 2)
    best_beta = best_x[-1]
    
    # Recalculate centroid and S for final output
    centroid = np.mean(best_pts, axis=0)
    
    def calc_S_final(beta, pts, centroid):
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

    best_S = calc_S_final(best_beta, best_pts, centroid)
    
    # Prepare output
    inner = []
    angle_deg = math.degrees(best_beta)
    for x, y in best_pts:
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(centroid[0]), float(centroid[1])]
    side = float(best_S)
    rotation = angle_deg
    
    return inner, center, side, rotation
