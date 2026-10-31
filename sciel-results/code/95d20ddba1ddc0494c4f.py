import numpy as np
from scipy.optimize import linprog
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def grid_to_cart(q, r):
        return (np.sqrt(3) * (q + r / 2.0), 1.5 * r)
        
    core_grid = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    
    # Pattern 1: Cardinals in shell 2
    p1_extra = [(2,0), (0,2), (-2,0), (0,-2)]
    p1_grid = core_grid + p1_extra
    
    # Pattern 2: Diagonals in shell 2
    p2_extra = [(2,-1), (1,-2), (-1,2), (-2,1)]
    p2_grid = core_grid + p2_extra
    
    # Pattern 3: Mixed
    p3_extra = [(2,-1), (1,-2), (0,-2), (-1,-1)]
    p3_grid = core_grid + p3_extra
    
    # Pattern 4: Linear
    p4_grid = []
    for i in range(-5, 6):
        p4_grid.append((i * np.sqrt(3), 0.0))
        
    patterns = [
        [grid_to_cart(q, r) for q, r in p1_grid],
        [grid_to_cart(q, r) for q, r in p2_grid],
        [grid_to_cart(q, r) for q, r in p3_grid],
        p4_grid
    ]
    
    best_S = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_points = patterns[0]
    
    def solve_lp(points, theta):
        n_k = []
        for k in range(6):
            angle = theta + k * np.pi / 3.0
            n_k.append((np.cos(angle), np.sin(angle)))
        
        A = []
        b = []
        
        for px, py in points:
            for nx, ny in n_k:
                A.append([-nx, -ny, -np.sqrt(3)/2])
                b.append(-(px*nx + py*ny))
                A.append([nx, ny, -np.sqrt(3)/2])
                b.append(px*nx + py*ny)
                
        A = np.array(A)
        b = np.array(b)
        c = np.array([0, 0, 1])
        
        try:
            res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method='highs')
            if res.success:
                return res.fun, (res.x[0], res.x[1])
            else:
                return float('inf'), (0,0)
        except Exception:
            return float('inf'), (0,0)

    for pts in patterns:
        if time.time() > deadline:
            break
        for deg in range(0, 61, 1):
            if time.time() > deadline:
                break
            theta = np.deg2rad(deg)
            S, center = solve_lp(pts, theta)
            if S < best_S:
                best_S = S
                best_center = center
                best_angle = deg
                best_points = pts
                
    if time.time() < deadline - 10:
        for deg in range(int(best_angle) - 2, int(best_angle) + 3):
             if deg < 0 or deg > 60: continue
             theta = np.deg2rad(deg)
             S, center = solve_lp(best_points, theta)
             if S < best_S:
                 best_S = S
                 best_center = center
                 best_angle = deg
                 
    inner = []
    for x, y in best_points:
        inner.append([x, y, 30.0])
        
    return inner, best_center, best_S, best_angle
# EVOLVE-BLOCK-END
