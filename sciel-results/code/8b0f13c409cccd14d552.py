import numpy as np
from scipy.optimize import minimize
import time

N = 11

def get_hex_vertices(cx, cy, angle, side=1.0):
    base_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    verts = verts @ R.T
    verts = verts * side
    verts[:, 0] += cx
    verts[:, 1] += cy
    return verts

def check_inside(verts, L):
    x = verts[:, 0]
    y = verts[:, 1]
    c1 = np.abs(y) <= (np.sqrt(3)/2) * L
    c2 = np.abs(np.sqrt(3)*x + y) <= np.sqrt(3) * L
    c3 = np.abs(-np.sqrt(3)*x + y) <= np.sqrt(3) * L
    return np.all(c1 & c2 & c3)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = 40.0
    best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    best_center = (0.0, 0.0)
    best_side = 40.0
    best_angle = 0.0

    # Initial guess: Hexagonal grid
    # Row 0: 2
    # Row 1: 3
    # Row 2: 4
    # Row 3: 2
    centers = []
    rows = [2, 3, 4, 2]
    y_pos = 0.0
    for i, count in enumerate(rows):
        x_start = -(count - 1) * np.sqrt(3) / 2
        for j in range(count):
            x = x_start + j * np.sqrt(3)
            centers.append((x, y_pos))
        y_pos += 1.5
        
    x0 = np.array([c[0] for c in centers])
    y0 = np.array([c[1] for c in centers])
    angle0 = np.zeros(N)
    L0 = 6.0
    
    x_init = np.concatenate([x0, y0, angle0, [L0]])

    def objective(params):
        x = params[0:N]
        y = params[N:2*N]
        angles = params[2*N:3*N]
        L = params[3*N]
        
        penalty = 0.0
        for i in range(N):
            verts = get_hex_vertices(x[i], y[i], angles[i])
            vx = verts[:, 0]
            vy = verts[:, 1]
            
            v1 = np.abs(vy) - (np.sqrt(3)/2) * L
            v2 = np.abs(np.sqrt(3)*vx + vy) - np.sqrt(3) * L
            v3 = np.abs(-np.sqrt(3)*vx + vy) - np.sqrt(3) * L
            
            penalty += np.sum(np.maximum(0, v1)) + np.sum(np.maximum(0, v2)) + np.sum(np.maximum(0, v3))
            
        return L + 100.0 * penalty

    for restart in range(10):
        if time.time() > deadline:
            break
            
        if restart == 0:
            x_curr = x_init.copy()
        else:
            x_curr = x_init.copy()
            x_curr[0:2*N] += np.random.normal(0, 0.5, 2*N)
            x_curr[2*N:3*N] += np.random.uniform(0, np.pi, N)
            x_curr[3*N] = 6.0

        # Stage 1: Nelder-Mead
        res = minimize(objective, x_curr, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.fun < best_score:
            # Stage 2: SLSQP Polish
            res2 = minimize(objective, res.x, method='SLSQP', 
                            options={'maxiter': 500, 'ftol': 1e-6})
            
            if res2.fun < res.fun:
                res = res2
            
            L_opt = res.x[3*N]
            x_opt = res.x[0:N]
            y_opt = res.x[N:2*N]
            angle_opt = res.x[2*N:3*N]
            
            valid = True
            for i in range(N):
                verts = get_hex_vertices(x_opt[i], y_opt[i], angle_opt[i])
                if not check_inside(verts, L_opt):
                    valid = False
                    break
            
            if valid:
                if L_opt < best_score:
                    best_score = L_opt
                    best_inner = [[x_opt[i], y_opt[i], np.rad2deg(angle_opt[i])] for i in range(N)]
                    best_center = (0.0, 0.0)
                    best_side = L_opt
                    best_angle = 0.0

    return best_inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-START
# (Function defined above)
# EVOLVE-BLOCK-END
