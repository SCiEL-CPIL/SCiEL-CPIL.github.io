"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Lattice vectors for side length 1 hexagons
    d = np.sqrt(3)
    
    # Define initial lattice positions for 11 hexagons
    init_pos = [(0.0, 0.0)]
    for k in range(6):
        theta = k * np.pi / 3
        init_pos.append((d * np.cos(theta), d * np.sin(theta)))
    for k in range(4):
        theta = k * np.pi / 3
        init_pos.append((2 * d * np.cos(theta), 2 * d * np.sin(theta)))
        
    while len(init_pos) < N:
        k = len(init_pos) - 7
        theta = (k + 4) * np.pi / 3
        init_pos.append((2 * d * np.cos(theta), 2 * d * np.sin(theta)))
        
    init_pos = init_pos[:N]

    x0 = np.zeros(3 * N)
    
    def get_vertices(params):
        verts_list = []
        for i in range(N):
            dx = params[3*i]
            dy = params[3*i+1]
            ang = params[3*i+2]
            
            cx = init_pos[i][0] + dx
            cy = init_pos[i][1] + dy
            
            angles = ang + np.arange(6) * np.pi / 3
            vx = cx + np.cos(angles)
            vy = cy + np.sin(angles)
            verts = np.column_stack([vx, vy])
            verts_list.append(verts)
            
        return verts_list

    def compute_outer_side(verts_list):
        apothem = np.sqrt(3) / 2
        S = 0.0
        normal_angles = np.pi/6 + np.arange(6) * np.pi/3
        
        for verts in verts_list:
            for k in range(6):
                n_x = np.cos(normal_angles[k])
                n_y = np.sin(normal_angles[k])
                projections = verts[:, 0] * n_x + verts[:, 1] * n_y
                max_proj = np.max(np.abs(projections))
                S = max(S, max_proj / apothem)
                
        return S

    def sat_overlap_depth(v1, v2):
        polys = [v1, v2]
        min_depth = np.inf
        for poly in polys:
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm == 0: continue
                normal = normal / norm
                
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                
                if max1 < min2 or max2 < min1:
                    return 0.0
                
                overlap = min(max1, max2) - max(min1, min2)
                if overlap < min_depth:
                    min_depth = overlap
                    
        return min_depth

    def overlap_penalty(verts_list):
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                depth = sat_overlap_depth(verts_list[i], verts_list[j])
                if depth > 0:
                    penalty += depth ** 2
        return penalty

    def objective(params):
        verts_list = get_vertices(params)
        S = compute_outer_side(verts_list)
        pen = overlap_penalty(verts_list)
        return S + 10.0 * pen

    bounds = []
    for i in range(N):
        bounds.append((-1.5, 1.5))
        bounds.append((-1.5, 1.5))
        bounds.append((-np.pi/6, np.pi/6))
        
    best_params = x0
    best_score = objective(x0)
    
    np.random.seed(42)
    for i in range(5):
        if i == 0:
            p = x0
        else:
            p = x0 + np.random.normal(0, 0.1, size=x0.size)
            
        res = minimize(objective, p, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-8})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
    inner = []
    for i in range(N):
        dx = best_params[3*i]
        dy = best_params[3*i+1]
        ang = best_params[3*i+2]
        
        cx = init_pos[i][0] + dx
        cy = init_pos[i][1] + dy
        
        inner.append([cx, cy, np.degrees(ang)])
        
    center = (0.0, 0.0)
    side = best_score
    angle_deg = 0.0
    
    return inner, center, side, angle_deg
# EVOLVE-BLOCK-END
