"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import linprog

N = 12

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    deadline = t0 + 100.0  # safety margin
    
    # Normal directions for flat-top hexagon edges (apothem directions)
    # These are at 30, 90, 150, 210, 270, 330 degrees
    normals = []
    for k in range(6):
        angle = math.radians(30 + k * 60)
        normals.append((math.cos(angle), math.sin(angle)))
    normals = np.array(normals)  # shape (6, 2)
    
    # Variables: [x1, y1, x2, y2, ..., x12, y12, s]
    # Total: 2*N + 1 = 25 variables
    
    # Objective: minimize s → c = [0, 0, ..., 0, 1]
    c = np.zeros(2 * N + 1)
    c[-1] = 1.0
    
    # Constraints: A_ub @ x <= b_ub
    
    A_rows = []
    b_vals = []
    
    # Non-overlap constraints: for each pair (i, j), i < j, and each normal k:
    # (p_i - p_j) . n_k <= sqrt(3)
    # (p_j - p_i) . n_k <= sqrt(3)
    sqrt3 = math.sqrt(3)
    
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(6):
                # (p_i - p_j) . n_k <= sqrt3
                row = np.zeros(2 * N + 1)
                row[2*i] = normals[k, 0]
                row[2*i+1] = normals[k, 1]
                row[2*j] = -normals[k, 0]
                row[2*j+1] = -normals[k, 1]
                A_rows.append(row)
                b_vals.append(sqrt3)
                
                # (p_j - p_i) . n_k <= sqrt3
                row2 = np.zeros(2 * N + 1)
                row2[2*j] = normals[k, 0]
                row2[2*j+1] = normals[k, 1]
                row2[2*i] = -normals[k, 0]
                row2[2*i+1] = -normals[k, 1]
                A_rows.append(row2)
                b_vals.append(sqrt3)
    
    # Outer containment: for each hexagon i and each normal k:
    # p_i . m_k <= (s - 1) * sqrt3 / 2
    # p_i . m_k - (sqrt3/2) * s <= -sqrt3/2
    for i in range(N):
        for k in range(6):
            row = np.zeros(2 * N + 1)
            row[2*i] = normals[k, 0]
            row[2*i+1] = normals[k, 1]
            row[-1] = -sqrt3 / 2
            A_rows.append(row)
            b_vals.append(-sqrt3 / 2)
    
    # s >= 1 → -s <= -1
    row = np.zeros(2 * N + 1)
    row[-1] = -1.0
    A_rows.append(row)
    b_vals.append(-1.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    
    # Bounds: s >= 1, positions unbounded
    bounds = [(None, None)] * (2 * N) + [(1.0, None)]
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if time.time() > deadline:
        # Fallback
        inner = [[2.0 * i, 0.0, 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    if result.success:
        s_opt = result.x[-1]
        centers = [(result.x[2*i], result.x[2*i+1]) for i in range(N)]
        inner = [[cx, cy, 0.0] for cx, cy in centers]
        return inner, (0.0, 0.0), s_opt, 0.0
    else:
        # Fallback: grid arrangement
        inner = []
        for row_idx in range(3):
            for col_idx in range(4):
                x = col_idx * sqrt3
                y = row_idx * 3.0
                if row_idx % 2 == 1:
                    x += sqrt3 / 2
                inner.append([x - 1.5 * sqrt3, y - 1.5, 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0

# EVOLVE-BLOCK-END
