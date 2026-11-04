import numpy as np
from scipy.optimize import minimize
import time

N = 12

def solve_packing(points, deadline):
    points = np.array(points)
    center = np.mean(points, axis=0)
    points = points - center
    
    best_L = float('inf')
    best_rotation = 0.0
    best_translation = np.array([0.0, 0.0])
    best_rotated_points = points
    
    normals = np.array([
        [np.cos(np.pi/3 * k), np.sin(np.pi/3 * k)] for k in range(6)
    ])
    
    angles = np.linspace(0, np.pi/6, 30)
    
    for phi in angles:
        if time.time() > deadline:
            break
        R = np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi), np.cos(phi)]
        ])
        rotated_points = points @ R.T
        
        proj = rotated_points @ normals.T
        C_plus = np.max(proj, axis=0)
        C_minus = -np.min(proj, axis=0)
        
        def trans_obj(t):
            tx, ty = t
            max_val = 0
            for k in range(6):
                nx, ny = normals[k]
                val1 = C_plus[k] + tx*nx + ty*ny
                val2 = C_minus[k] - tx*nx - ty*ny
                m = val1 if val1 > val2 else val2
                if m > max_val:
                    max_val = m
            return max_val
        
        res = minimize(trans_obj, [0.0, 0.0], method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if res.success:
            L_val = (2/np.sqrt(3)) * res.fun
            if L_val < best_L:
                best_L = L_val
                best_rotation = phi
                best_translation = res.x
                best_rotated_points = rotated_points + np.array(best_translation)
        else:
            L_val = (2/np.sqrt(3)) * trans_obj([0.0, 0.0])
            if L_val < best_L:
                best_L = L_val
                best_rotation = phi
                best_translation = [0.0, 0.0]
                best_rotated_points = rotated_points

    return best_L, best_rotation, best_rotated_points

def construct():
    start_time = time.time()
    deadline = start_time + 120.0 # Safety margin
    
    d = np.sqrt(3)
    
    # Shape 1: 3x4 Grid
    p1 = []
    for r in range(3):
        for c in range(4):
            x = c * d
            y = r * 1.5
            if r % 2 == 1:
                x += d / 2
            p1.append([x, y])
            
    # Shape 2: 2x6 Grid
    p2 = []
    for r in range(2):
        for c in range(6):
            x = c * d
            y = r * 1.5
            if r % 2 == 1:
                x += d / 2
            p2.append([x, y])
            
    # Shape 3: 2-3-4-3
    p3 = []
    p3.append([-d/2, 0])
    p3.append([d/2, 0])
    p3.append([-d, 1.5])
    p3.append([0, 1.5])
    p3.append([d, 1.5])
    p3.append([-1.5*d, 3.0])
    p3.append([-0.5*d, 3.0])
    p3.append([0.5*d, 3.0])
    p3.append([1.5*d, 3.0])
    p3.append([-d, 4.5])
    p3.append([0, 4.5])
    p3.append([d, 4.5])
    
    shapes = [p1, p2, p3]
    
    best_overall_L = float('inf')
    best_overall_rot = 0.0
    best_overall_pts = None
    
    for i, shape in enumerate(shapes):
        if time.time() > deadline:
            break
        L, rot, pts = solve_packing(shape, deadline)
        if L < best_overall_L:
            best_overall_L = L
            best_overall_rot = rot
            best_overall_pts = pts
            
    if best_overall_pts is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for p in best_overall_pts:
        inner.append([p[0], p[1], np.degrees(best_overall_rot)])
        
    return inner, (0.0, 0.0), best_overall_L, 0.0
