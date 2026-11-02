import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    sqrt3 = np.sqrt(3)
    
    best_side = float('inf')
    best_center = None
    best_theta = None
    best_positions = None
    
    # Define honeycomb arrangements: list of row counts
    # For flat-top hexagons in honeycomb:
    # - Rows are spaced sqrt(3)/2 apart vertically
    # - Within a row, centers are 2 apart horizontally
    # - Odd rows offset by 1 horizontally
    
    explicit_configs = [
        [3, 4, 4],
        [4, 3, 4],
        [2, 4, 5],
        [5, 4, 2],
        [1, 5, 5],
        [5, 5, 1],
        [3, 5, 3],
        [5, 3, 3],
        [4, 4, 3],
        [3, 3, 5],
        [2, 5, 4],
        [4, 5, 2],
        [1, 4, 6],
        [6, 4, 1],
        [2, 3, 6],
        [6, 3, 2],
        [1, 3, 7],
        [7, 3, 1],
        [3, 1, 7],
        [7, 1, 3],
        [1, 7, 3],
        [3, 7, 1],
        [4, 7, 0],  # invalid, skip
    ]
    
    for row_counts in explicit_configs:
        if time.time() > deadline - 15:
            break
        if sum(row_counts) != N or any(c <= 0 for c in row_counts):
            continue
        
        positions = []
        for i, count in enumerate(row_counts):
            y = i * sqrt3 / 2
            x_start = -(count - 1) / 2.0
            for j in range(count):
                x = x_start + j
                if i % 2 == 1:
                    x += 0.5
                positions.append((x, y))
        
        if len(positions) != N:
            continue
        
        pos_arr = np.array(positions)
        
        # Try multiple rotations of the enclosing hexagon
        for theta_deg in range(0, 60, 5):  # 0, 5, 10, ..., 55
            if time.time() > deadline - 15:
                break
            theta = np.radians(theta_deg)
            angles = theta + np.arange(6) * np.pi / 3
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            
            def objective(c):
                diffs = pos_arr - c
                projs = np.abs(diffs @ normals.T)
                return projs.max()
            
            c0 = pos_arr.mean(axis=0)
            try:
                res = minimize(objective, c0, method='Nelder-Mead',
                              options={'xatol': 1e-9, 'fatol': 1e-11, 'maxiter': 300})
                max_proj = res.fun
                s = 2.0 / sqrt3 * max_proj
                if s < best_side:
                    best_side = s
                    best_center = res.x
                    best_theta = theta
                    best_positions = positions
            except:
                pass
    
    # Fallback if optimization failed
    if best_positions is None:
        positions = [(2.0 * k, 0.0) for k in range(N)]
        pos_arr = np.array(positions)
        c = pos_arr.mean(axis=0)
        x_range = pos_arr[:, 0].max() - pos_arr[:, 0].min()
        s = x_range / sqrt3
        best_side = s
        best_center = c
        best_theta = 0.0
        best_positions = positions
    
    inner = [[float(x), float(y), float(np.degrees(best_theta))] for (x, y) in best_positions]
    center = (float(best_center[0]), float(best_center[1]))
    side = float(best_side)
    angle = float(np.degrees(best_theta))
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END

