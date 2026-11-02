"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11
import numpy as np
from scipy.optimize import minimize
import itertools
import time

# EVOLVE-BLOCK-START
SQRT3 = np.sqrt(3)
NORMALS = np.array([
    [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))]
    for a in range(0, 360, 60)
])
VERTEX_OFFSETS = np.array([
    [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))]
    for a in range(0, 360, 60)
])

def run_optimization(initial_coords, time_limit=10.0):
    start = time.time()
    max_extent = 0
    for i in range(N):
        cx, cy = initial_coords[i]
        for v_off in VERTEX_OFFSETS:
            px = cx + v_off[0]
            py = cy + v_off[1]
            for n in NORMALS:
                dist = abs(px * n[0] + py * n[1])
                if dist > max_extent:
                    max_extent = dist
    L_init = max_extent * 2.0 / SQRT3 + 0.5
    
    x0 = np.concatenate([initial_coords.flatten(), [L_init]])
    
    def objective(x):
        return x[2*N]
        
    def constraint_disjoint(x):
        coords = x[:2*N].reshape(N, 2)
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                cons.append(dx*dx + dy*dy - 3.0)
        return np.array(cons)
        
    def constraint_containment(x):
        coords = x[:2*N].reshape(N, 2)
        L = x[2*N]
        h = L * SQRT3 / 2.0
        cons = []
        for i in range(N):
            cx, cy = coords[i]
            for v_off in VERTEX_OFFSETS:
                px = cx + v_off[0]
                py = cy + v_off[1]
                for n in NORMALS:
                    dist = px * n[0] + py * n[1]
                    cons.append(h - dist)
                    cons.append(h + dist)
        return np.array(cons)
        
    constraints = [
        {'type': 'ineq', 'fun': constraint_disjoint},
        {'type': 'ineq', 'fun': constraint_containment}
    ]
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-4}
        )
        
        if res.success and time.time() - start < time_limit:
            return res.x[2*N], res.x[:2*N].reshape(N, 2)
    except Exception:
        pass
        
    return np.inf, initial_coords

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    base_coords = [
        [0.0, 0.0],
        [SQRT3, 0.0],
        [SQRT3/2, 1.5],
        [-SQRT3/2, 1.5],
        [-SQRT3, 0.0],
        [-SQRT3/2, -1.5],
        [SQRT3/2, -1.5]
    ]
    
    extra_dirs = [
        [SQRT3, 0.0],
        [SQRT3/2, 1.5],
        [-SQRT3/2, 1.5],
        [-SQRT3, 0.0],
        [-SQRT3/2, -1.5],
        [SQRT3/2, -1.5]
    ]
    
    best_L = np.inf
    best_inner = None
    
    # Try a few symmetric layouts
    layouts_to_try = [
        [0, 1, 3, 4], # Right, Up-Right, Up-Left, Left
        [0, 2, 4, 5], # Right, Up-Left, Left, Down-Left
        [1, 2, 3, 4], # Up-Right, Up-Left, Left, Down-Left
        [0, 1, 2, 3], # Right, Up-Right, Up-Left, Left
    ]
    
    for combo in layouts_to_try:
        if time.time() > deadline:
            break
            
        coords = [c.copy() for c in base_coords]
        for idx in combo:
            new_x = base_coords[idx+1][0] + extra_dirs[idx][0]
            new_y = base_coords[idx+1][1] + extra_dirs[idx][1]
            coords.append([new_x, new_y])
            
        L, final_coords = run_optimization(np.array(coords))
        
        if L < best_L:
            best_L = L
            best_inner = final_coords
            
    if best_inner is None:
        # Fallback
        best_inner = np.array(base_coords + [[SQRT3*2, 0], [SQRT3, 3], [-SQRT3, 3], [-SQRT3*2, 0]])
        max_extent = 0
        for i in range(N):
            cx, cy = best_inner[i]
            for v_off in VERTEX_OFFSETS:
                px = cx + v_off[0]
                py = cy + v_off[1]
                for n in NORMALS:
                    dist = abs(px * n[0] + py * n[1])
                    if dist > max_extent:
                        max_extent = dist
        best_L = max_extent * 2.0 / SQRT3
        
    inner = []
    for i in range(N):
        inner.append([float(best_inner[i][0]), float(best_inner[i][1]), 0.0])
        
    return inner, (0.0, 0.0), float(best_L), 0.0
# EVOLVE-BLOCK-END
