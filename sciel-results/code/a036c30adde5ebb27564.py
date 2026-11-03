import numpy as np
import time

N = 11

def min_enclosing_hex_side(points, cx, cy, angle_deg):
    theta = np.deg2rad(angle_deg)
    normals = theta + np.pi/6 + np.arange(6) * np.pi/3
    inradius_factor = np.sqrt(3) / 2
    max_proj = 0.0
    for px, py in points:
        dx = px - cx
        dy = py - cy
        projs = dx * np.cos(normals) + dy * np.sin(normals)
        max_proj = max(max_proj, np.max(projs))
    return max_proj / inradius_factor

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    sqrt3 = np.sqrt(3)
    
    # Hex grid: axial (q,r) -> (x,y)
    # x = sqrt3 * (q + r/2), y = 1.5 * r
    
    # Try several arrangements
    arrangements = [
        # 7-hex cluster + 4 extensions
        [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1), (2,0), (2,-1), (0,-2), (-2,1)],
        [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1), (2,0), (0,-2), (-2,1), (-2,2)],
        # 3-4-4
        [(0,0), (1,0), (-1,0), (0,1), (1,1), (-1,1), (0,-1), (1,-1), (-1,-1), (2,0), (2,-1)],
        # Linear-ish
        [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,-1), (1,-1), (2,-1), (-1,0), (-1,1)],
    ]
    
    best_side = float('inf')
    best_arr = None
    best_cx = 0.0
    best_cy = 0.0
    best_angle = 0.0
    
    for arr in arrangements:
        if len(arr) != N:
            continue
        points = [(sqrt3 * (q + r/2), 1.5 * r) for q, r in arr]
        
        # Center
        cx = np.mean([p[0] for p in points])
        cy = np.mean([p[1] for p in points])
        
        for angle in [0, 30, 60, 90, 120, 150]:
            side = min_enclosing_hex_side(points, cx, cy, angle)
            if side < best_side:
                best_side = side
                best_arr = arr
                best_cx = cx
                best_cy = cy
                best_angle = angle
    
    if best_arr is None:
        inner = [[sqrt3 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = []
    for q, r in best_arr:
        x = sqrt3 * (q + r/2)
        y = 1.5 * r
        inner.append([x, y, 0.0])
    
    return inner, (best_cx, best_cy), best_side, best_angle
