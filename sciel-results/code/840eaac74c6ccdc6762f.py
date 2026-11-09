import time
import math
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    sqrt3 = math.sqrt(3.0)
    
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y))
    
    unit_verts = np.array([
        [math.cos(math.radians(i*60)), math.sin(math.radians(i*60))] 
        for i in range(6)
    ])

    def compute_L(centers, alpha):
        ca = math.cos(alpha)
        sa = math.sin(alpha)
        R = np.array([[ca, -sa], [sa, ca]])
        rot_verts = unit_verts @ R.T
        verts = centers[:, np.newaxis, :] + rot_verts[np.newaxis, :, :]
        verts = verts.reshape(-1, 2)
        
        normals = np.array([
            [math.cos(math.radians(30 + i*60)), math.sin(math.radians(30 + i*60))] 
            for i in range(6)
        ])
        
        projs = verts @ normals.T
        max_projs = projs.max(axis=0)
        min_projs = projs.min(axis=0)
        
        widths = max_projs - min_projs
        L = np.max(widths) / sqrt3
        
        return L

    best_L = float('inf')
    best_subset = None
    best_alpha = 0.0
    
    angles_to_test = [0.0, math.pi/12, math.pi/6]
    
    for subset in combinations(range(19), 12):
        if time.time() > deadline - 10.0:
            break
            
        centers = np.array([all_centers[i] for i in subset])
        
        for alpha in angles_to_test:
            L = compute_L(centers, alpha)
            if L < best_L:
                best_L = L
                best_subset = subset
                best_alpha = alpha

    if best_subset is None:
        return ([], (0,0), 100.0, 0.0)

    init_centers = np.array([all_centers[i] for i in best_subset])
    
    # Optimize alpha
    res_alpha = minimize(lambda a: compute_L(init_centers, a[0]), [best_alpha], method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-10})
    if res_alpha.success:
        best_alpha = res_alpha.x[0]
        best_L = res_alpha.fun

    # Optimize positions
    def objective_pos(params):
        if time.time() > deadline:
            return 1e10
        centers = params.reshape(12, 2)
        L = compute_L(centers, best_alpha)
        
        dists = np.sqrt(np.sum((centers[:, np.newaxis] - centers[np.newaxis, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        
        if min_d < sqrt3:
            penalty = 100.0 * (sqrt3 - min_d)**2
        else:
            penalty = 0.0
            
        return L + penalty

    x0 = init_centers.flatten()
    res_pos = minimize(objective_pos, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
    
    if res_pos.success and res_pos.fun < best_L:
        best_L = res_pos.fun
        init_centers = res_pos.x.reshape(12, 2)

    # Finalize
    ca = math.cos(best_alpha)
    sa = math.sin(best_alpha)
    R = np.array([[ca, -sa], [sa, ca]])
    rot_verts = unit_verts @ R.T
    verts = init_centers[:, np.newaxis, :] + rot_verts[np.newaxis, :, :]
    verts = verts.reshape(-1, 2)
    
    center = np.mean(verts, axis=0)
    
    # Calculate actual required L for this center
    normals = np.array([
        [math.cos(math.radians(30 + i*60)), math.sin(math.radians(30 + i*60))] 
        for i in range(6)
    ])
    rel_verts = verts - center
    projs = rel_verts @ normals.T
    max_proj = np.max(projs)
    actual_L = 2.0 * max_proj / sqrt3
    
    # Use the larger of the optimized L and the actual required L to ensure validity
    final_L = max(best_L, actual_L)
    
    inner = [[init_centers[i][0], init_centers[i][1], math.degrees(best_alpha)] for i in range(12)]
    outer_center = (center[0], center[1])
    outer_side = final_L
    outer_angle = 0.0
    
    return (inner, outer_center, outer_side, outer_angle)
