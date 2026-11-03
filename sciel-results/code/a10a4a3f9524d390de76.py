import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0

    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1),
        (-1, 0), (0, -1), (1, -1)
    ]
    coords.append((2, 0))
    coords.append((-2, 0))
    coords.append((1, 1))
    coords.append((-1, -1))
    
    x0 = []
    for q, r in coords:
        x, y = axial_to_cartesian(q, r)
        x0.extend([x, y, 0.0])
    x0 = np.array(x0)

    def get_hex_vertices(cx, cy, angle, side=1.0):
        angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + angle)
        vs = np.zeros((6, 2))
        vs[:, 0] = cx + side * np.cos(angles)
        vs[:, 1] = cy + side * np.sin(angles)
        return vs

    def calc_S_and_overlap(x):
        max_A = 0.0
        overlap = 0.0
        hex_centers = []
        all_vertices = []
        
        for i in range(N):
            cx = x[3*i]
            cy = x[3*i+1]
            ang = x[3*i+2]
            hex_centers.append((cx, cy))
            vs = get_hex_vertices(cx, cy, ang)
            all_vertices.extend(vs)
            
        all_vertices = np.array(all_vertices)
        
        best_A = float('inf')
        for outer_ang in np.linspace(0, 60, 7):
            base_angs = np.deg2rad([30, 90, 150]) + np.deg2rad(outer_ang)
            normals = np.column_stack((np.cos(base_angs), np.sin(base_angs)))
            dists = np.abs(all_vertices @ normals.T)
            A_req = np.max(dists)
            if A_req < best_A:
                best_A = A_req
                
        S = 2.0 * best_A / np.sqrt(3)
        
        # Overlap check
        min_dist = np.sqrt(3) * 1.001
        for i in range(N):
            for j in range(i+1, N):
                dx = hex_centers[i][0] - hex_centers[j][0]
                dy = hex_centers[i][1] - hex_centers[j][1]
                d = np.sqrt(dx*dx + dy*dy)
                if d < min_dist:
                    overlap += (min_dist - d)**2
                    
        return S + 100.0 * overlap, best_A, S

    def objective(x):
        val, _, _ = calc_S_and_overlap(x)
        return val

    best_x = x0
    best_val, _, best_S = calc_S_and_overlap(x0)
    
    # Optimize
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'disp': False})
    
    if time.time() < deadline - 10:
        if res.fun < best_val:
            best_x = res.x
            best_val, _, best_S = calc_S_and_overlap(best_x)
            
    # Check validity of best_x
    # If overlap > 0, fallback to x0
    val, _, _ = calc_S_and_overlap(best_x)
    if val > best_S: # Heuristic: if penalty is significant
         # Check actual overlap
         overlap_part = val - best_S
         if overlap_part > 1e-6:
             best_x = x0
             best_val, _, best_S = calc_S_and_overlap(x0)

    inner = []
    for i in range(N):
        inner.append([best_x[3*i], best_x[3*i+1], np.rad2deg(best_x[3*i+2])])
        
    # Find best outer rotation
    all_vertices = []
    for i in range(N):
        cx = best_x[3*i]
        cy = best_x[3*i+1]
        ang = best_x[3*i+2]
        vs = get_hex_vertices(cx, cy, ang)
        all_vertices.extend(vs)
    all_vertices = np.array(all_vertices)
    
    best_A = float('inf')
    best_outer_ang = 0.0
    for outer_ang in np.linspace(0, 60, 101):
        base_angs = np.deg2rad([30, 90, 150]) + np.deg2rad(outer_ang)
        normals = np.column_stack((np.cos(base_angs), np.sin(base_angs)))
        dists = np.abs(all_vertices @ normals.T)
        A_req = np.max(dists)
        if A_req < best_A:
            best_A = A_req
            best_outer_ang = outer_ang
            
    final_S = 2.0 * best_A / np.sqrt(3)
    
    return inner, (0.0, 0.0), final_S, best_outer_ang
# EVOLVE-BLOCK-END
