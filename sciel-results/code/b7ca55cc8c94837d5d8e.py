import numpy as np
import scipy.optimize as opt
import time

N = 11

def hex_support(center, angle, dir_angle):
    dx = np.cos(dir_angle)
    dy = np.sin(dir_angle)
    alpha = dir_angle - angle
    alpha_mod = alpha % (np.pi / 3.0)
    h = (np.sqrt(3) / 2.0) / np.cos(alpha_mod - np.pi / 6.0)
    center_proj = center[0] * dx + center[1] * dy
    return center_proj + h

def check_disjoint(hex1, hex2):
    x1, y1, a1 = hex1
    x2, y2, a2 = hex2
    
    def get_vertices(x, y, a):
        angles = a + np.arange(6) * np.pi / 3.0
        return np.column_stack((x + np.cos(angles), y + np.sin(angles)))
    
    v1 = get_vertices(x1, y1, a1)
    v2 = get_vertices(x2, y2, a2)
    
    def get_axes(v):
        axes = []
        for i in range(6):
            p1 = v[i]
            p2 = v[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n = n / np.linalg.norm(n)
            axes.append(n)
        return axes
    
    axes1 = get_axes(v1)
    axes2 = get_axes(v2)
    all_axes = axes1 + axes2
    
    for axis in all_axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            return True
    return False

def calculate_S(hexes, phi):
    max_support = 0.0
    for k in range(6):
        dir_angle = phi + k * np.pi / 3.0
        h_dir = 0.0
        for h in hexes:
            val = hex_support(h, dir_angle)
            if val > h_dir:
                h_dir = val
        if h_dir > max_support:
            max_support = h_dir
    return (2.0 / np.sqrt(3.0)) * max_support

def objective(params, deadline):
    if time.time() > deadline:
        return 1e10
    hexes = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        hexes.append((x, y, a))
    phi = params[33]
    
    S = calculate_S(hexes, phi)
    
    for i in range(N):
        for j in range(i+1, N):
            dx = hexes[i][0] - hexes[j][0]
            dy = hexes[i][1] - hexes[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1.7:
                if not check_disjoint(hexes[i], hexes[j]):
                    return S + 1000.0
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial config
    a1 = np.array([np.sqrt(3), 0])
    a2 = np.array([np.sqrt(3)/2, 1.5])
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * a1 + j * a2
            if np.linalg.norm(p) < 4.0:
                points.append(p)
    points.sort(key=lambda p: np.linalg.norm(p))
    
    init_hexes = []
    for p in points[:11]:
        init_hexes.append((p[0], p[1], 0.0))
        
    init_params = []
    for h in init_hexes:
        init_params.extend([h[0], h[1], h[2]])
    init_params.append(0.0)
    init_params = np.array(init_params)
    
    best_score = float('inf')
    best_params = init_params
    
    for restart in range(20):
        if time.time() > deadline:
            break
        if restart > 0:
            noise = np.random.normal(0, 0.1, size=init_params.shape)
            p = init_params + noise
        else:
            p = init_params.copy()
            
        res = opt.minimize(lambda x: objective(x, deadline), p, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
    # Extract
    hexes_final = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        a_deg = np.degrees(a) % 360.0
        hexes_final.append([x, y, a_deg])
        
    phi = best_params[33]
    phi_deg = np.degrees(phi) % 360.0
    
    # Verify
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not check_disjoint(hexes_final[i], hexes_final[j]):
                valid = False
                break
        if not valid: break
        
    if not valid:
        # Fallback to initial
        hexes_final = [[h[0], h[1], 0.0] for h in init_hexes]
        phi_deg = 0.0
        best_score = calculate_S(init_hexes, 0.0)
        
    return hexes_final, (0.0, 0.0), best_score, phi_deg
