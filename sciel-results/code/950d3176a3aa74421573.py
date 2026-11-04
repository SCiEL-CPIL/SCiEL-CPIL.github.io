import numpy as np
import scipy.optimize as opt

def construct():
    N = 12
    
    centers = []
    for i in range(-1, 2):
        for j in range(-1, 3):
            x = i * np.sqrt(3)
            y = 2 * j + (1 if i % 2 != 0 else 0)
            centers.append([x, y])
    centers = np.array(centers)
    centers -= np.mean(centers, axis=0)
    
    angles = np.deg2rad([90, 150, 210, 270, 330, 30])
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    def calc_L(params):
        rot_deg, cx, cy = params
        rot = np.deg2rad(rot_deg)
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        
        max_L = 0
        for c_center in centers:
            rc = R @ c_center
            rc[0] += cx
            rc[1] += cy
            
            rv = (R @ verts.T).T + rc
            
            x = rv[:, 0]
            y = rv[:, 1]
            
            L1 = np.abs(y)
            L2 = (2 / np.sqrt(3)) * np.abs(x)
            L3 = (2 / np.sqrt(3)) * np.abs(y + x / np.sqrt(3))
            L4 = (2 / np.sqrt(3)) * np.abs(y - x / np.sqrt(3))
            
            L_max = np.max(np.maximum(np.maximum(L1, L2), np.maximum(L3, L4)))
            if L_max > max_L:
                max_L = L_max
                
        return max_L

    # Initial guess from coarse search
    best_L = float('inf')
    best_params = [0, 0, 0]
    
    for rot in range(0, 180, 5):
        L = calc_L([rot, 0, 0])
        if L < best_L:
            best_L = L
            best_params = [rot, 0, 0]
            
    # Refine with scipy
    res = opt.minimize(calc_L, best_params, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000})
    
    best_rot, best_cx, best_cy = res.x
    best_L = res.fun
    
    rot_rad = np.deg2rad(best_rot)
    c, s = np.cos(rot_rad), np.sin(rot_rad)
    R = np.array([[c, -s], [s, c]])
    
    inner = []
    for c_center in centers:
        rc = R @ c_center
        rc[0] += best_cx
        rc[1] += best_cy
        inner.append([rc[0], rc[1], best_rot])
        
    return inner, (best_cx, best_cy), best_L, 0.0
