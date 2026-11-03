import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
MIN_DIST = math.sqrt(3)
EPS_GAP = 1e-6

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def calc_metrics(params):
        pts = np.array(params[:2*N]).reshape(N, 2)
        theta = params[2*N]
        cx = params[2*N + 1]
        cy = params[2*N + 2]
        
        # Calculate S
        max_proj = 0.0
        for k in range(6):
            ang = theta + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        
        S = (max_proj + APOTHEM) / APOTHEM
        
        # Calculate penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < MIN_DIST + EPS_GAP:
                    penalty += (MIN_DIST + EPS_GAP - dist) ** 2
        
        return S, penalty

    def objective(params):
        S, penalty = calc_metrics(params)
        return S + 100.0 * penalty

    # Generate initial candidates
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Hexagonal Lattice
    lattice_pts = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            lattice_pts.append([x, y])
    lattice_pts = np.array(lattice_pts)
    lattice_pts -= np.mean(lattice_pts, axis=0)
    dists = np.linalg.norm(lattice_pts, axis=1)
    idx = np.argsort(dists)[:N]
    init1 = lattice_pts[idx].flatten().tolist()
    init1.extend([0.0, 0.0, 0.0])
    inits.append(init1)
    
    # 2. Random
    for _ in range(10):
        pts = rng.uniform(-3, 3, size=(N, 2))
        pts -= np.mean(pts, axis=0)
        init_r = pts.flatten().tolist()
        init_r.extend([rng.uniform(0, math.pi/6), 0.0, 0.0])
        inits.append(init_r)
        
    best_S = float('inf')
    best_params = None
    
    # Evaluate baseline
    S0, p0 = calc_metrics(inits[0])
    if p0 < 1e-9:
        best_S = S0
        best_params = inits[0]
        
    for init in inits[1:]:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        try:
            res = minimize(objective, init, method='Powell', options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-8})
            S_val, p_val = calc_metrics(res.x)
            if p_val < 1e-9:
                if S_val < best_S:
                    best_S = S_val
                    best_params = res.x
        except Exception:
            pass

    if best_params is None:
        best_params = inits[0]
        best_S, _ = calc_metrics(best_params)

    pts = np.array(best_params[:2*N]).reshape(N, 2)
    theta = best_params[2*N]
    cx = best_params[2*N + 1]
    cy = best_params[2*N + 2]
    
    # Recalculate S for final answer
    max_proj = 0.0
    for k in range(6):
        ang = theta + math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(ang)
        ny = math.sin(ang)
        projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    S_final = (max_proj + APOTHEM) / APOTHEM
    
    inner = []
    angle_deg = math.degrees(theta)
    for x, y in pts:
        inner.append([float(x + cx), float(y + cy), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
