import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    sqrt3 = np.sqrt(3)
    
    # Initial configuration: 4-3-4 grid of pointy-top hexagons
    centers = []
    # Row 0 (y=0): 4 hexagons
    for i in range(4):
        x = (i - 1.5) * sqrt3
        centers.append([x, 0.0, 0.0])
    # Row 1 (y=1.5): 3 hexagons
    for i in range(3):
        x = (i - 1.0) * sqrt3
        centers.append([x, 1.5, 0.0])
    # Row 2 (y=3.0): 4 hexagons
    for i in range(4):
        x = (i - 1.5) * sqrt3
        centers.append([x, 3.0, 0.0])
        
    def get_hex_vertices(cx, cy, angle, side=1.0):
        # Pointy-top vertices
        angles = np.deg2rad(30 + np.arange(6) * 60 + angle)
        vx = cx + side * np.cos(angles)
        vy = cy + side * np.sin(angles)
        return np.column_stack((vx, vy))

    def objective(vars):
        inner_vars = vars[:3*N].reshape(N, 3)
        Xc, Yc, phi, S = vars[3*N:3*N+4]
        
        if S <= 0:
            return 1e10
            
        phi_rad = np.deg2rad(phi)
        normals = []
        for k in range(6):
            theta = phi_rad + k * np.pi / 3.0
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        apothem = S * np.sqrt(3) / 2.0
        
        max_violation = 0.0
        for i in range(N):
            cx, cy, ang = inner_vars[i]
            verts = get_hex_vertices(cx, cy, ang, 1.0)
            diffs = verts - np.array([Xc, Yc])
            projections = diffs @ normals.T
            violations = np.max(np.abs(projections) - apothem)
            if violations > max_violation:
                max_violation = violations
                
        overlap_penalty = 0.0
        min_dist = np.sqrt(3)
        for i in range(N):
            for j in range(i+1, N):
                dx = inner_vars[i, 0] - inner_vars[j, 0]
                dy = inner_vars[i, 1] - inner_vars[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    overlap_penalty += (min_dist - dist)**2
                    
        return S + 10.0 * max_violation + 10.0 * overlap_penalty

    def opt_outer(inner_fixed):
        def obj_outer(vars):
            Xc, Yc, phi, S = vars
            if S <= 0: return 1e10
            phi_rad = np.deg2rad(phi)
            normals = []
            for k in range(6):
                theta = phi_rad + k * np.pi / 3.0
                normals.append([np.cos(theta), np.sin(theta)])
            normals = np.array(normals)
            apothem = S * np.sqrt(3) / 2.0
            max_violation = 0.0
            for i in range(N):
                cx, cy, ang = inner_fixed[i]
                verts = get_hex_vertices(cx, cy, ang, 1.0)
                diffs = verts - np.array([Xc, Yc])
                projections = diffs @ normals.T
                violations = np.max(np.abs(projections) - apothem)
                if violations > max_violation:
                    max_violation = violations
            return S + 10.0 * max_violation

        x0_outer = np.array([0.0, 1.5, 0.0, 4.0])
        res_outer = minimize(obj_outer, x0_outer, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        return res_outer.x

    outer_init = opt_outer(centers)
    
    x0_full = np.array([c[0] for c in centers] + 
                       [c[1] for c in centers] + 
                       [c[2] for c in centers] + 
                       outer_init)
                       
    best_result = x0_full
    best_score = objective(x0_full)
    
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        jitter = 0.1
        x0_jitter = x0_full.copy()
        for i in range(3*N):
            x0_jitter[i] += np.random.uniform(-jitter, jitter)
        x0_jitter[3*N+3] += np.random.uniform(-0.2, 0.2)
        
        res = minimize(objective, x0_jitter, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-7})
        
        if res.fun < best_score:
            best_score = res.fun
            best_result = res.x
            
    inner_vars = best_result[:3*N].reshape(N, 3)
    Xc, Yc, phi, S = best_result[3*N:3*N+4]
    
    inner = [[inner_vars[i, 0], inner_vars[i, 1], inner_vars[i, 2]] for i in range(N)]
    
    return inner, (Xc, Yc), S, phi

# EVOLVE-BLOCK-END
