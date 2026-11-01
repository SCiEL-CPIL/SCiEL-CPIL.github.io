import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 4.0, 0.0

    # Generate 11 points on a triangular lattice
    # Center (0,0) + 6 in ring 1 + 4 in ring 2
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    pts = np.array(pts)
    
    # The inner hexagons have side length 1, apothem = sqrt(3)/2
    # The outer hexagon has side length S, apothem = S * sqrt(3)/2
    # For an inner hexagon centered at p with orientation beta, 
    # the support function in direction n is (p-c).n + apothem_inner
    # This must be <= apothem_outer = S * apothem_inner
    # So (p-c).n <= S*apothem_inner - apothem_inner = (S-1)*apothem_inner
    # Let D = (S-1)*apothem_inner, minimize D, then S = D/apothem_inner + 1
    
    def compute_S(beta, cx, cy):
        """Compute the minimum S for given beta, cx, cy"""
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        max_proj = 0.0
        for nx, ny in normals:
            for px, py in pts:
                proj = (px - cx) * nx + (py - cy) * ny
                max_proj = max(max_proj, abs(proj))
        
        D = max_proj
        S = D / APOTHEM + 1.0
        return S
    
    def objective(params):
        beta, cx, cy = params
        return compute_S(beta, cx, cy)
    
    # Try multiple starting points
    best_S = float('inf')
    best_params = None
    
    # The optimal beta is likely 0 or pi/6 due to symmetry
    # Start with beta=0 and optimize cx, cy
    for beta0 in [0.0, math.pi/6, math.pi/3]:
        for cx0 in [0.0]:
            for cy0 in [0.0]:
                try:
                    res = minimize(objective, [beta0, cx0, cy0], method='Nelder-Mead',
                                   options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_params = res.x
                except Exception:
                    pass
    
    if best_params is None:
        # Fallback
        best_S = 3.5
        best_params = [0.0, 0.0, 0.0]
    
    beta, cx, cy = best_params
    
    # Verify and adjust for strict disjointness
    # Scale points slightly to ensure gaps
    SCALE = 1.0 + 1e-7
    pts_scaled = []
    for x, y in pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    pts_scaled = np.array(pts_scaled)
    
    # Recompute S for scaled points
    max_proj = 0.0
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        for px, py in pts_scaled:
            proj = abs((px - cx) * nx + (py - cy) * ny)
            max_proj = max(max_proj, proj)
    
    S = max_proj / APOTHEM + 1.0
    
    angle_deg = math.degrees(beta)
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
