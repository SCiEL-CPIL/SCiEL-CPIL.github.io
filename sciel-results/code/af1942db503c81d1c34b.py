import numpy as np
from scipy.optimize import minimize
import time

N = 12
MIN_DIST = np.sqrt(3)

angles = np.deg2rad(30 + np.arange(6) * 60)
normals = np.column_stack((np.cos(angles), np.sin(angles)))

def get_vertices(center):
    base_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    return center + base_verts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Initial guess: Hexagonal lattice
    points = []
    max_q = 5
    for q in range(-max_q, max_q + 1):
        for r in range(-max_q, max_q + 1):
            x = 1.5 * q
            y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
            dist = np.sqrt(x*x + y*y)
            points.append((dist, x, y))
            
    points.sort(key=lambda x: x[0])
    init_points = points[:N]
    x0 = np.array([p[1] for p in init_points])
    y0 = np.array([p[2] for p in init_points])
    
    max_dist = np.max(np.sqrt(x0**2 + y0**2))
    S0 = max_dist + 1.5
    
    def objective(vars):
        if time.time() > deadline:
            return 1e10
        x = vars[:2*N].reshape(N, 2)
        S = vars[2*N]
        
        # Containment penalty
        pen_cont = 0.0
        for i in range(N):
            verts = get_vertices(x[i])
            proj = verts @ normals.T
            max_proj = np.max(proj, axis=0)
            limit = np.sqrt(3)/2 * S
            viol = np.maximum(0, max_proj - limit)
            pen_cont += np.sum(viol**2)
            
        # Overlap penalty
        pen_ov = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(x[i] - x[j])
                if d < MIN_DIST:
                    pen_ov += (MIN_DIST - d)**2
                    
        return S + 50.0 * pen_cont + 50.0 * pen_ov

    x0_flat = x0.flatten()
    y0_flat = y0.flatten()
    init_vars = np.concatenate([x0_flat, y0_flat, [S0]])
    
    bounds = [(None, None)] * (2*N) + [(3.0, 20.0)]
    
    best_S = S0
    best_vars = init_vars.copy()
    
    # Run 1
    try:
        res = minimize(objective, init_vars, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 3000, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.fun < best_S:
            best_S = res.x[2*N]
            best_vars = res.x
    except Exception:
        pass

    # Run 2 (Polish)
    if time.time() < deadline - 5:
        try:
            res2 = minimize(objective, best_vars, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-8})
            if res2.fun < best_S:
                best_S = res2.x[2*N]
                best_vars = res2.x
        except Exception:
            pass
            
    final_x = best_vars[:2*N].reshape(N, 2)
    final_S = best_vars[2*N]
    
    # Fallback if optimization failed
    if final_S > 40.0:
        final_S = 40.0
        final_x = np.column_stack([4.0 * np.arange(-6, 6), np.zeros(12)])

    inner = []
    for i in range(N):
        inner.append([final_x[i, 0], final_x[i, 1], 0.0])
        
    return inner, (0.0, 0.0), final_S, 0.0
