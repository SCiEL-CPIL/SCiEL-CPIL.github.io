import math
import numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations

N = 11
APOTHEM = math.sqrt(3) / 2.0

def construct():
    # Central 7 points
    central_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
    ]
    # 6 candidate outer points (distance 3)
    outer_candidates_axial = [
        (1, 1), (-1, 2), (-2, 1), (-2, -1), (1, -2), (2, -1)
    ]
    
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
        
    central_pts = np.array([axial_to_cartesian(q, r) for q, r in central_axial])
    outer_candidates = [axial_to_cartesian(q, r) for q, r in outer_candidates_axial]
    
    best_S = float('inf')
    best_config = None
    best_beta = 0.0
    
    # Try all combinations of 4 outer points
    for combo in combinations(range(6), 4):
        outer_pts = np.array([outer_candidates[i] for i in combo])
        pts = np.vstack([central_pts, outer_pts])
        
        centroid = np.mean(pts, axis=0)
        
        def calc_S(beta):
            normals = []
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            max_req_S = 0.0
            for nx, ny in normals:
                rel_pts = pts - centroid
                projections = rel_pts[:, 0] * nx + rel_pts[:, 1] * ny
                abs_proj = np.max(np.abs(projections))
                required_S = (abs_proj + APOTHEM) / APOTHEM
                if required_S > max_req_S:
                    max_req_S = required_S
            return max_req_S

        res = minimize_scalar(calc_S, bounds=(0, math.pi / 6), method='bounded')
        if res.fun < best_S:
            best_S = res.fun
            best_config = pts
            best_beta = res.x
            best_centroid = centroid

    # Prepare output
    inner = []
    angle_deg = math.degrees(best_beta)
    for x, y in best_config:
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(best_centroid[0]), float(best_centroid[1])]
    side = float(best_S)
    rotation = angle_deg
    
    return inner, center, side, rotation
