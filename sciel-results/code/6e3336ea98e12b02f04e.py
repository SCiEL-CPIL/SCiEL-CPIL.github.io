import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0

    d_min = math.sqrt(3)
    
    # Initial configuration: "Flower" + 4 in ring 2
    pts = []
    pts.append((0, 0))
    for i in range(6):
        ang = i * math.pi / 3
        pts.append((d_min * math.cos(ang), d_min * math.sin(ang)))
        
    ring2_4 = [(2,0), (0,2), (-2,0), (0,-2)]
    for q,r in ring2_4:
        x = d_min * (q + r/2.0)
        y = d_min * (math.sqrt(3)/2.0) * r
        pts.append((x,y))
        
    init_pts = np.array(pts)
    init_thetas = np.zeros(N)
    
    x0 = np.zeros(N*3)
    for i in range(N):
        x0[i*3] = init_pts[i, 0]
        x0[i*3+1] = init_pts[i, 1]
        x0[i*3+2] = init_thetas[i]
        
    verts_rel = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])
    
    def calc_S_and_penalty(x):
        pts = x.reshape(N, 3)
        cx = pts[:, 0]
        cy = pts[:, 1]
        thetas = pts[:, 2]
        
        all_points = []
        for i in range(N):
            c, s = math.cos(thetas[i]), math.sin(thetas[i])
            R = np.array([[c, -s], [s, c]])
            verts_rot = verts_rel @ R.T
            verts_abs = verts_rot + np.array([cx[i], cy[i]])
            all_points.append(verts_abs)
            
        all_points = np.vstack(all_points)
        
        k_angles = np.arange(6) * math.pi / 3
        normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
        
        dots = np.sum(normals[:, np.newaxis, :] * all_points[np.newaxis, :, :], axis=-1)
        abs_dots = np.abs(dots)
        max_per_normal = np.max(abs_dots, axis=-1)
        max_proj = np.max(max_per_normal)
        
        S = (2.0 / math.sqrt(3)) * max_proj
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < d_min:
                    penalty += (d_min - dist)**2
                    
        return S + 1000.0 * penalty
    
    best_x = x0
    best_val = calc_S_and_penalty(x0)
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - safety_margin:
            break
            
        if r == 0:
            x_start = x0
        else:
            x_start = x0 + np.random.normal(0, 0.5, size=N*3)
            x_start[2::3] = np.random.uniform(-0.2, 0.2, size=N)
            
        try:
            res = minimize(calc_S_and_penalty, x_start, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            continue
            
    best_pts = best_x.reshape(N, 3)
    cx = best_pts[:, 0]
    cy = best_pts[:, 1]
    thetas = best_pts[:, 2]
    
    def get_S_for_outer_rot(theta_outer):
        all_points = []
        for i in range(N):
            c, s = math.cos(thetas[i]), math.sin(thetas[i])
            R = np.array([[c, -s], [s, c]])
            verts_rot = verts_rel @ R.T
            verts_abs = verts_rot + np.array([cx[i], cy[i]])
            all_points.append(verts_abs)
        all_points = np.vstack(all_points)
        
        k_angles = np.arange(6) * math.pi / 3 + theta_outer
        normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
        
        dots = np.sum(normals[:, np.newaxis, :] * all_points[np.newaxis, :, :], axis=-1)
        abs_dots = np.abs(dots)
        max_per_normal = np.max(abs_dots, axis=-1)
        max_proj = np.max(max_per_normal)
        
        return (2.0 / math.sqrt(3)) * max_proj
        
    best_S = float('inf')
    best_theta_outer = 0
    for k in range(6):
        theta_out = k * math.pi / 6
        S_val = get_S_for_outer_rot(theta_out)
        if S_val < best_S:
            best_S = S_val
            best_theta_outer = theta_out
            
    inner_rows = []
    for i in range(N):
        inner_rows.append([cx[i], cy[i], math.degrees(thetas[i])])
        
    center = (0.0, 0.0)
    side = best_S
    angle_deg = math.degrees(best_theta_outer)
    
    return inner_rows, center, side, angle_deg
