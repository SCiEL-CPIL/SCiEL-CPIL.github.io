import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
MIN_DIST = 2.0 * APOTHEM  # sqrt(3)
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def calc_enclosing_hex_size(points, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        projs = points @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10
            
        pts = params[:2*N].reshape(N, 2)
        beta = params[2*N]
        
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < MIN_DIST:
                    overlap_penalty += (MIN_DIST - d) ** 2
        
        if overlap_penalty > 1e-6:
            return 1e6 * overlap_penalty + calc_enclosing_hex_size(pts, beta)
            
        return calc_enclosing_hex_size(pts, beta)

    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    starts = []
    
    # 1. 7 cluster + 4 corners
    base = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    extra1 = [(2,0), (0,2), (-2,1), (0,-2)]
    pts1 = [axial_to_cartesian(q, r) for q, r in base + extra1]
    starts.append(pts1)
    
    # 2. 7 cluster + 4 mid-edges
    extra2 = [(1,1), (-1,2), (-1,-1), (1,-2)]
    pts2 = [axial_to_cartesian(q, r) for q, r in base + extra2]
    starts.append(pts2)
    
    # 3. 3x3 block + 2
    base4 = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)]
    extra4 = [(3,1), (-1,1)]
    pts4 = [axial_to_cartesian(q, r) for q, r in base4 + extra4]
    starts.append(pts4)
    
    # 4. Random perturbation of 1
    pts5 = [ (p[0] + np.random.normal(0, 0.2), p[1] + np.random.normal(0, 0.2)) for p in pts1 ]
    starts.append(pts5)
    
    # 5. Random perturbation of 2
    pts6 = [ (p[0] + np.random.normal(0, 0.2), p[1] + np.random.normal(0, 0.2)) for p in pts2 ]
    starts.append(pts6)

    best_S = float('inf')
    best_params = None

    for start_pts in starts:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        init_params = np.array(start_pts).flatten()
        init_params = np.append(init_params, 0.0)
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    best_pts = best_params[:2*N].reshape(N, 2)
    best_beta = best_params[2*N]
    
    # Verify validity
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < MIN_DIST - 1e-9:
                valid = False
    
    if not valid:
        # Fallback to first start
        init_params = np.array(starts[0]).flatten()
        init_params = np.append(init_params, 0.0)
        res = minimize(objective, init_params, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
        best_pts = res.x[:2*N].reshape(N, 2)
        best_beta = res.x[2*N]

    S_final = calc_enclosing_hex_size(best_pts, best_beta)
    angle_deg = math.degrees(best_beta)
    
    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [0.0, 0.0]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
