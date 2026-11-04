import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # 1. Generate the 12 hexagon centers in a 3x4 honeycomb cluster
    d = np.sqrt(3)
    centers = []
    for i in range(3):
        for j in range(4):
            x = j * d + (i % 2) * (d / 2)
            y = i * 1.5
            centers.append((x, y))
    
    # Center the cluster at origin
    cx = np.mean([c[0] for c in centers])
    cy = np.mean([c[1] for c in centers])
    centers = [(c[0] - cx, c[1] - cy) for c in centers]
    
    # 2. Generate vertices for each hexagon
    angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    
    points = []
    for cx, cy in centers:
        for k in range(6):
            px = cx + cos_a[k]
            py = cy + sin_a[k]
            points.append([px, py])
            
    points = np.array(points)
    
    # 3. Define the normals for the outer hexagon
    n_angles = np.array([np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6, 3*np.pi/2, 11*np.pi/6])
    normals = np.array([[np.cos(a), np.sin(a)] for a in n_angles])
    
    # 4. Function to compute required side length S for a given rotation theta
    def min_side(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        # Rotate points
        # points is (72, 2)
        # We want (72, 2) @ (2, 2).T
        # x' = x c - y s
        # y' = x s + y c
        x = points[:, 0]
        y = points[:, 1]
        rx = x * c - y * s
        ry = x * s + y * c
        
        # Project onto normals
        # normals is (6, 2)
        # proj = rx * n_x + ry * n_y
        # We want max abs of these 6 projections for all 72 points.
        
        # Vectorized:
        # (72, 1) * (1, 6) + (72, 1) * (1, 6) -> (72, 6)
        proj_x = rx[:, np.newaxis] * normals[:, 0][np.newaxis, :]
        proj_y = ry[:, np.newaxis] * normals[:, 1][np.newaxis, :]
        proj = proj_x + proj_y
        
        max_abs = np.max(np.abs(proj))
        return (2.0 / np.sqrt(3.0)) * max_abs

    # 5. Optimize theta
    n_grid = 5000
    thetas = np.linspace(0, np.pi/3, n_grid)
    best_s = np.inf
    best_theta = 0
    
    # Batch evaluation for speed
    # We can't easily batch the min_side function as written without refactoring, 
    # but 5000 calls is fine.
    
    for th in thetas:
        s_val = min_side(th)
        if s_val < best_s:
            best_s = s_val
            best_theta = th
            
    if time.time() > deadline - 5:
        return _format_result(centers, best_theta, best_s)

    # Refine
    step = (np.pi/3) / n_grid
    lo = max(0, best_theta - step)
    hi = min(np.pi/3, best_theta + step)
    
    for _ in range(100):
        if time.time() > deadline - 2:
            break
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        s1 = min_side(m1)
        s2 = min_side(m2)
        if s1 < s2:
            hi = m2
        else:
            lo = m1
            
    best_theta = (lo + hi) / 2
    best_s = min_side(best_theta)
    
    return _format_result(centers, best_theta, best_s)

def _format_result(centers, theta, side):
    centers_arr = np.array(centers)
    c = np.cos(theta)
    s = np.sin(theta)
    rot_matrix = np.array([[c, -s], [s, c]])
    global_centers = centers_arr @ rot_matrix.T
    
    inner = []
    angle_deg = np.degrees(theta)
    for i in range(12):
        x, y = global_centers[i]
        inner.append([x, y, angle_deg])
        
    outer_center = [0.0, 0.0]
    outer_side = side
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
