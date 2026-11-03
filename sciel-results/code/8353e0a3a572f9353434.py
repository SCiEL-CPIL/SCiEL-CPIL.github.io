"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    sqrt3 = math.sqrt(3)
    
    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = np.array([
            [cx + math.cos(angle_rad + i * math.pi / 3),
             cy + math.sin(angle_rad + i * math.pi / 3)]
            for i in range(6)
        ])
        return verts
    
    def enclosing_hexagon_side(verts, cx, cy, angle_deg):
        theta = math.radians(angle_deg)
        max_proj = 0.0
        for k in range(6):
            a = theta + k * math.pi / 3
            nx = math.cos(a)
            ny = math.sin(a)
            proj = (verts[:, 0] - cx) * nx + (verts[:, 1] - cy) * ny
            max_proj = max(max_proj, proj.max())
        return 2.0 * max_proj / sqrt3
    
    def make_arrangement(row_counts, gap=0.0):
        """Create honeycomb arrangement with given row counts."""
        positions = []
        for r, ncol in enumerate(row_counts):
            for c in range(ncol):
                x = c * (sqrt3 + gap) + (r % 2) * (sqrt3 + gap) / 2
                y = r * (1.5 + gap * 0.5)
                positions.append((x, y))
        return positions
    
    # Generate candidate arrangements
    arrangements = []
    
    # Various row patterns summing to 12
    patterns = [
        [3, 4, 3, 2],
        [2, 4, 4, 2],
        [4, 4, 4],
        [3, 4, 4, 1],
        [1, 4, 4, 3],
        [2, 5, 5],
        [5, 4, 3],
        [3, 5, 4],
        [4, 5, 3],
        [2, 3, 4, 3],
        [3, 3, 3, 3],
        [1, 3, 5, 3],
        [3, 5, 3, 1],
        [2, 2, 4, 4],
        [4, 4, 2, 2],
    ]
    
    for p in patterns:
        if sum(p) == N:
            arrangements.append(make_arrangement(p))
    
    # Also try a hexagonal cluster: center + 6 + 5
    arr_hex = [(0, 0)]
    for k in range(6):
        a = k * math.pi / 3
        arr_hex.append((sqrt3 * math.cos(a), sqrt3 * math.sin(a)))
    for k in range(5):
        a = k * math.pi / 3
        arr_hex.append((2 * sqrt3 * math.cos(a), 2 * sqrt3 * math.sin(a)))
    arrangements.append(arr_hex)
    
    # Evaluate each arrangement
    best_side = 1e10
    best_outer = None
    best_arr = None
    
    for arr in arrangements:
        if time.time() > deadline - 10:
            break
        if len(arr) != N:
            continue
        
        all_verts = np.vstack([hex_vertices(x, y, 0.0) for x, y in arr])
        com = all_verts.mean(axis=0)
        
        def objective(params):
            cx, cy, angle = params
            return enclosing_hexagon_side(all_verts, cx, cy, angle)
        
        best_local = 1e10
        best_params_local = None
        
        for init_angle in [0, 30, 60]:
            if time.time() > deadline - 10:
                break
            x0 = np.array([com[0], com[1], init_angle])
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
                if res.fun < best_local:
                    best_local = res.fun
                    best_params_local = res.x.copy()
            except Exception:
                continue
        
        if best_local < best_side:
            best_side = best_local
            best_outer = best_params_local
            best_arr = arr
    
    if best_outer is None:
        arr = arrangements[0]
        all_verts = np.vstack([hex_vertices(x, y, 0.0) for x, y in arr])
        com = all_verts.mean(axis=0)
        best_outer = np.array([com[0], com[1], 0.0])
        best_side = enclosing_hexagon_side(all_verts, com[0], com[1], 0.0)
        best_arr = arr
    
    outer_x, outer_y, outer_angle = best_outer
    inner = [[x, y, 0.0] for x, y in best_arr]
    
    return inner, (outer_x, outer_y), best_side, outer_angle
# EVOLVE-BLOCK-END
