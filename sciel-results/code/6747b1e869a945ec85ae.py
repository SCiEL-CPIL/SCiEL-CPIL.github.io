import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.radians(np.arange(6) * 60)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def find_min_enclosing_hex(inner_pos, inner_ang, inner_side=1.0):
    """Return (side, center_x, center_y, angle_deg) of minimal enclosing regular hexagon."""
    n = len(inner_pos)
    all_verts = []
    for i in range(n):
        verts = hexagon_vertices(inner_pos[i][0], inner_pos[i][1], inner_side, inner_ang[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    best_s = np.inf
    best_cx, best_cy, best_enc_ang = 0, 0, 0
    
    for enc_deg in np.arange(0, 60, 1.0):
        enc_rad = np.radians(enc_deg)
        normals = enc_rad + np.radians(np.arange(6) * 60)
        
        min_p = np.zeros(6)
        max_p = np.zeros(6)
        for k in range(6):
            d = np.array([np.cos(normals[k]), np.sin(normals[k])])
            proj = all_verts @ d
            min_p[k] = proj.min()
            max_p[k] = proj.max()
        
        centroid = all_verts.mean(axis=0)
        
        best_a = np.inf
        best_cx_k, best_cy_k = centroid[0], centroid[1]
        for dx in np.arange(-4, 4.01, 0.25):
            for dy in np.arange(-4, 4.01, 0.25):
                cx, cy = centroid[0] + dx, centroid[1] + dy
                a = 0.0
                for k in range(6):
                    nx, ny = np.cos(normals[k]), np.sin(normals[k])
                    proj = cx * nx + cy * ny
                    a = max(a, proj - min_p[k], max_p[k] - proj)
                if a < best_a:
                    best_a = a
                    best_cx_k, best_cy_k = cx, cy
        
        s = best_a / (np.sqrt(3) / 2)
        if s < best_s:
            best_s = s
            best_cx, best_cy, best_enc_ang = best_cx_k, best_cy_k, enc_deg
    
    return best_s, best_cx, best_cy, best_enc_ang

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    sqrt3 = np.sqrt(3)
    hex_coords = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1), (2,0), (-2,0), (0,2), (0,-2)]
    positions = []
    for q, r in hex_coords:
        x = sqrt3 * (q + r/2)
        y = 1.5 * r
        positions.append([x, y])
    
    inner_ang = [0.0] * N
    s0, cx0, cy0, ang0 = find_min_enclosing_hex(positions, inner_ang)
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        pos = [[params[3*i], params[3*i+1]] for i in range(N)]
        angs = [params[3*i+2] for i in range(N)]
        s, _, _, _ = find_min_enclosing_hex(pos, angs)
        return s
    
    x0 = np.zeros(3*N)
    for i in range(N):
        x0[3*i] = positions[i][0]
        x0[3*i+1] = positions[i][1]
        x0[3*i+2] = 0.0
    
    best_s = s0
    best_params = x0.copy()
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-3, 'fatol': 1e-3})
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x
    except:
        pass
    
    if time.time() > deadline - 20:
        pos = [[best_params[3*i], best_params[3*i+1]] for i in range(N)]
        angs = [best_params[3*i+2] for i in range(N)]
        s, cx, cy, enc_ang = find_min_enclosing_hex(pos, angs)
        inner = [[pos[i][0], pos[i][1], angs[i]] for i in range(N)]
        return inner, (cx, cy), s, enc_ang
    
    try:
        res2 = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-4})
        if res2.fun < best_s:
            best_s = res2.fun
            best_params = res2.x
    except:
        pass
    
    pos = [[best_params[3*i], best_params[3*i+1]] for i in range(N)]
    angs = [best_params[3*i+2] for i in range(N)]
    s, cx, cy, enc_ang = find_min_enclosing_hex(pos, angs)
    inner = [[pos[i][0], pos[i][1], angs[i]] for i in range(N)]
    return inner, (cx, cy), s, enc_ang
