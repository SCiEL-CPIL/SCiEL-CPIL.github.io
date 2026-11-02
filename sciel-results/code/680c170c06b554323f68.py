"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    sqrt3 = np.sqrt(3)
    
    # Axial to Cartesian for flat-top hexagons
    def axial_to_cartesian(q, r):
        x = 2.0 * q + r
        y = sqrt3 * r
        return (x, y)
    
    # 12 positions on hex grid: center + ring1 (6) + 5 from ring2
    axial_positions = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2), (-1, -1),
    ]
    
    positions = [axial_to_cartesian(q, r) for q, r in axial_positions]
    
    # Compute minimal enclosing hexagon
    def compute_min_enclosing_hex(positions):
        A_rows = []
        b_vals = []
        
        for (xi, yi) in positions:
            vertices = [
                (xi + 1, yi),
                (xi + 0.5, yi + sqrt3 / 2),
                (xi - 0.5, yi + sqrt3 / 2),
                (xi - 1, yi),
                (xi - 0.5, yi - sqrt3 / 2),
                (xi + 0.5, yi - sqrt3 / 2),
            ]
            for (px, py) in vertices:
                A_rows.append([-1, 0, -1])
                b_vals.append(-px)
                A_rows.append([1, 0, -1])
                b_vals.append(px)
                A_rows.append([-0.5, -sqrt3/2, -1])
                b_vals.append(-px/2 - py*sqrt3/2)
                A_rows.append([0.5, sqrt3/2, -1])
                b_vals.append(px/2 + py*sqrt3/2)
                A_rows.append([-0.5, sqrt3/2, -1])
                b_vals.append(-px/2 + py*sqrt3/2)
                A_rows.append([0.5, -sqrt3/2, -1])
                b_vals.append(px/2 - py*sqrt3/2)
        
        A = np.array(A_rows)
        b = np.array(b_vals)
        c = np.array([0, 0, 1])
        
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method='highs')
        
        if res.success:
            return float(res.x[0]), float(res.x[1]), float(res.x[2])
        else:
            return None, None, float('inf')
    
    cx, cy, S = compute_min_enclosing_hex(positions)
    
    if cx is None or not np.isfinite(S):
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = [[float(pos[0]), float(pos[1]), 0.0] for pos in positions]
    
    return inner, (cx, cy), S, 0.0
# EVOLVE-BLOCK-END
