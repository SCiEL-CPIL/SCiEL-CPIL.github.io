import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0
    
    sqrt3 = np.sqrt(3.0)
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2.0, 1.5])
    
    # Star shape points in lattice coords (i, j)
    pts_lattice = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    
    pts = np.array(pts_lattice, dtype=float)
    
    P = np.zeros((len(pts), 2))
    P[:, 0] = pts[:, 0] * sqrt3 + pts[:, 1] * (sqrt3 / 2.0)
    P[:, 1] = pts[:, 1] * 1.5
    
    P -= P.mean(axis=0)
    
    best_S = float('inf')
    best_C = np.array([0.0, 0.0])
    best_R = 0.0
    best_P = P
    
    # Test 12 rotations (every 30 degrees)
    for k in range(12):
        angle = k * np.pi / 6.0
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        P_rot = P @ R.T
        
        step = 0.1
        min_coord = -3.0
        max_coord = 3.0
        xs = np.arange(min_coord, max_coord, step)
        ys = np.arange(min_coord, max_coord, step)
        
        for cx in xs:
            for cy in ys:
                X = P_rot[:, 0] + cx
                Y = P_rot[:, 1] + cy
                
                term1 = np.max(np.abs(X))
                term2 = (2.0 / sqrt3) * np.max(np.abs(Y))
                term3 = (1.0 / sqrt3) * np.max(np.abs(sqrt3 * X + Y))
                
                S = max(term1, term2, term3)
                
                if S < best_S:
                    best_S = S
                    best_C = np.array([cx, cy])
                    best_R = angle
                    best_P = P_rot
        
        if time.time() > deadline:
            break

    if time.time() < deadline - 10.0:
        step = 0.01
        min_coord = best_C[0] - 0.2
        max_coord = best_C[0] + 0.2
        xs = np.arange(min_coord, max_coord, step)
        
        min_coord_y = best_C[1] - 0.2
        max_coord_y = best_C[1] + 0.2
        ys = np.arange(min_coord_y, max_coord_y, step)
        
        P_refine = best_P
        for cx in xs:
            for cy in ys:
                X = P_refine[:, 0] + cx
                Y = P_refine[:, 1] + cy
                
                term1 = np.max(np.abs(X))
                term2 = (2.0 / sqrt3) * np.max(np.abs(Y))
                term3 = (1.0 / sqrt3) * np.max(np.abs(sqrt3 * X + Y))
                
                S = max(term1, term2, term3)
                
                if S < best_S:
                    best_S = S
                    best_C = np.array([cx, cy])

    inner = []
    angle_deg = best_R * 180.0 / np.pi
    for p in best_P:
        x = p[0] + best_C[0]
        y = p[1] + best_C[1]
        inner.append([x, y, angle_deg])
        
    center = [best_C[0], best_C[1]]
    side = best_S
    angle = 0.0
    
    return inner, center, side, angle
