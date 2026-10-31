import math
import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0

def construct():
    # Generate 11 points on a triangular lattice with slight expansion to prevent overlap
    # Axial coordinates (q, r)
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    # Scale factor slightly greater than 1 to ensure no overlap
    SCALE = 1.001
    
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0) * SCALE
        y = 1.5 * r * SCALE
        pts.append((x, y))
        
    pts = np.array(pts)
    
    # Calculate centroid
    centroid = np.mean(pts, axis=0)
    
    # Function to compute minimal side length S for a given outer rotation beta (radians)
    def calc_S(beta):
        # Check time limit
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        # Normals of the outer hexagon
        # Outer hexagon is centered at centroid, rotated by beta
        # Normals are at beta + pi/6 + k*pi/3
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        max_S = 0.0
        for nx, ny in normals:
            # Project all points onto this normal
            # Relative to centroid
            rel_pts = pts - centroid
            projections = rel_pts[:, 0] * nx + rel_pts[:, 1] * ny
            abs_proj = np.max(np.abs(projections))
            required_S = (abs_proj + APOTHEM) / APOTHEM
            if required_S > max_S:
                max_S = required_S
        
        return max_S

    # Minimize S over beta in [0, pi/6]
    try:
        res = minimize_scalar(calc_S, bounds=(0, math.pi / 6), method='bounded',
                             options={'maxiter': 100})
        best_beta = res.x
        best_S = res.fun
    except Exception:
        # Fallback: use beta = 0
        best_beta = 0.0
        best_S = calc_S(0.0)
    
    # Prepare output
    inner = []
    angle_deg = math.degrees(best_beta)
    for x, y in pts:
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(centroid[0]), float(centroid[1])]
    side = float(best_S)
    rotation = angle_deg
    
    return inner, center, side, rotation
