import numpy as np
import time
from scipy.optimize import linprog

N = 11

def check_no_overlap(positions, min_dist=2.0 - 1e-9):
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            if dx*dx + dy*dy < min_dist*min_dist:
                return False
    return True

def min_enclosing_hexagon_side(points, n_rotations=12):
    points = np.array(points)
    best_s = float('inf')
    best_center = None
    best_rot = 0
    
    for rot in range(n_rotations):
        angle = rot * np.pi / 6
        normals = np.array([[np.cos(angle + np.pi/6 + k*np.pi/3), 
                            np.sin(angle + np.pi/6 + k*np.pi/3)] for k in range(6)])
        
        m = len(points)
        A_ub = []
        b_ub = []
        for i in range(m):
            for j in range(6):
                A_ub.append([normals[j][0], normals[j][1], -np.sqrt(3)/2])
                b_ub.append(-float(points[i] @ normals[j]))
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        c_obj = [0, 0, 1]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, 
                         bounds=[(None,None),(None,None),(0,None)], 
                         method='highs')
            if res.success:
                s = res.x[2]
                if s < best_s:
                    best_s = s
                    best_center = (res.x[0], res.x[1])
                    best_rot = rot * 30
        except Exception:
            pass
    
    return best_s, best_center, best_rot

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    sqrt3 = np.sqrt(3)
    arrangements = []
    
    flower = [(0,0), (2,0), (1,sqrt3), (-1,sqrt3), (-2,0), (-1,-sqrt3), (1,-sqrt3)]
    
    extras_options = [
        [(3, sqrt3), (0, 2*sqrt3), (-3, sqrt3), (-3, -sqrt3)],
        [(3, sqrt3), (3, -sqrt3), (0, 2*sqrt3), (0, -2*sqrt3)],
        [(2, 2*sqrt3), (-2, 2*sqrt3), (2, -2*sqrt3), (-2, -2*sqrt3)],
        [(3, sqrt3), (0, 2*sqrt3), (-3, -sqrt3), (0, -2*sqrt3)],
        [(4, 0), (3, sqrt3), (3, -sqrt3), (0, 2*sqrt3)],
        [(4, 0), (3, sqrt3), (3, -sqrt3), (0, -2*sqrt3)],
        [(2, 2*sqrt3), (2, -2*sqrt3), (-2, 2*sqrt3), (-2, -2*sqrt3)],
    ]
    
    for extras in extras_options:
        arr = list(flower) + list(extras)
        if check_no_overlap(arr):
            arrangements.append(arr)
    
    # Row-based
    for n_extra_row in range(-2, 3):
        arr = []
        for x in [-2, 0, 2]:
            arr.append((x, 0))
        for x in [-3, -1, 1, 3]:
            arr.append((x, sqrt3))
        for x in [-2, 0, 2]:
            arr.append((x, 2*sqrt3))
        arr.append((n_extra_row, 3*sqrt3))
        if check_no_overlap(arr):
            arrangements.append(arr)
    
    arr = []
    for x in [-3, -1, 1, 3]:
        arr.append((x, 0))
    for x in [-2, 0, 2]:
        arr.append((x, sqrt3))
    for x in [-3, -1, 1, 3]:
        arr.append((x, 2*sqrt3))
    if check_no_overlap(arr):
        arrangements.append(arr)
    
    best_score = float('inf')
    best_result = None
    
    for arr in arrangements:
        if time.time() > deadline:
            break
        s, center, rot = min_enclosing_hexagon_side(arr, n_rotations=12)
        if s < best_score:
            best_score = s
            best_result = (arr, center, s, rot)
    
    if best_result is None:
        arr = list(flower) + [(3, sqrt3), (0, 2*sqrt3), (-3, sqrt3), (-3, -sqrt3)]
        s, center, rot = min_enclosing_hexagon_side(arr, n_rotations=12)
        best_result = (arr, center, s, rot)
    
    arr, center, s, rot = best_result
    inner = [[float(x), float(y), 0.0] for x, y in arr]
    return inner, center, float(s), float(rot)
