import numpy as np
import time
from scipy.optimize import minimize

N = 12

def get_hexagon_vertices(cx, cy, side, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    angles = angle_rad + np.deg2rad(np.arange(0, 360, 60))
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def point_distance_to_hexagon(px, py, cx, cy, side, angle_deg):
    """
    Returns 0 if inside, else distance to boundary.
    """
    dx = px - cx
    dy = py - cy
    angle_rad = -np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x_local = dx * cos_a - dy * sin_a
    y_local = dx * sin_a + dy * cos_a
    
    a = side * np.sqrt(3) / 2
    
    # Check 6 sides
    # 1. x + y/sqrt(3) <= side
    d1 = (x_local + y_local/np.sqrt(3)) - side
    # 2. y <= side * sqrt(3)/2
    d2 = y_local - a
    # 3. -x + y/sqrt(3) <= side
    d3 = (-x_local + y_local/np.sqrt(3)) - side
    # 4. -x - y/sqrt(3) <= side
    d4 = (-x_local - y_local/np.sqrt(3)) - side
    # 5. -y <= side * sqrt(3)/2
    d5 = -y_local - a
    # 6. x - y/sqrt(3) <= side
    d6 = (x_local - y_local/np.sqrt(3)) - side
    
    # If any d > 0, point is outside.
    # Distance is roughly the max positive d, scaled by normal length.
    # Normals lengths:
    # n1 = (1, 1/sqrt(3)), len = sqrt(1 + 1/3) = 2/sqrt(3)
    # n2 = (0, 1), len = 1
    # n3 = (-1, 1/sqrt(3)), len = 2/sqrt(3)
    # ...
    
    # To get true distance, we need to divide by normal length.
    # But for penalty, max(d1, 0) * (sqrt(3)/2) is a good approximation.
    # Let's just return max(d1, d2, d3, d4, d5, d6, 0)
    # This is proportional to distance.
    
    return max(d1, d2, d3, d4, d5, d6, 0.0)

def hexagon_overlap_penalty(v1, v2):
    edges1 = np.diff(np.vstack((v1, v1[0])), axis=0)
    edges2 = np.diff(np.vstack((v2, v2[0])), axis=0)
    normals1 = np.column_stack((edges1[:, 1], -edges1[:, 0]))
    normals2 = np.column_stack((edges2[:, 1], -edges2[:, 0]))
    norms1 = np.linalg.norm(normals1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(normals2, axis=1, keepdims=True)
    normals1 /= norms1
    normals2 /= norms2
    
    min_overlap = float('inf')
    all_normals = np.vstack((normals1, normals2))
    
    for normal in all_normals:
        proj1 = np.dot(v1, normal)
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = np.dot(v2, normal)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            return 0.0
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
    return min_overlap

def calculate_penalty(positions, S):
    penalty = 0.0
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    outer_side = S
    
    verts = []
    for i in range(N):
        x, y, ang = positions[i]
        v = get_hexagon_vertices(x, y, 1.0, ang)
        verts.append(v)
        for pt in v:
            dist = point_distance_to_hexagon(pt[0], pt[1], outer_center[0], outer_center[1], outer_side, outer_angle)
            if dist > 0:
                penalty += dist * dist

    for i in range(N):
        for j in range(i+1, N):
            pen = hexagon_overlap_penalty(verts[i], verts[j])
            if pen > 1e-6:
                penalty += pen * pen
                
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def get_initial_positions(S):
        pos = [[0.0, 0.0, 0.0]]
        for k in range(6):
            ang = np.deg2rad(k * 60)
            x = np.sqrt(3) * np.cos(ang)
            y = np.sqrt(3) * np.sin(ang)
            pos.append([x, y, 0.0])
        for k in range(5):
            ang = np.deg2rad(k * 60)
            x = 2 * np.sqrt(3) * np.cos(ang)
            y = 2 * np.sqrt(3) * np.sin(ang)
            pos.append([x, y, 0.0])
        return pos

    S_low = 3.5
    S_high = 5.0
    best_S = S_high
    best_pos = get_initial_positions(S_high)
    
    # Optimize S_high
    res = minimize(calculate_penalty, best_pos, args=(S_high,), method='L-BFGS-B', 
                   bounds=[(-S_high, S_high, 0, 360)]*N, options={'maxiter': 50})
    best_pos = res.x.reshape(N, 3)
    
    for _ in range(15):
        if time.time() > deadline:
            break
        S_mid = (S_low + S_high) / 2.0
        init_pos = best_pos.copy()
        
        res = minimize(calculate_penalty, init_pos, args=(S_mid,), method='L-BFGS-B', 
                       bounds=[(-S_mid, S_mid, 0, 360)]*N, options={'maxiter': 100, 'ftol': 1e-10})
        pos = res.x.reshape(N, 3)
        pen = calculate_penalty(pos, S_mid)
        
        if pen < 1e-4:
            S_high = S_mid
            best_S = S_mid
            best_pos = pos
        else:
            S_low = S_mid
            
    return best_pos, (0.0, 0.0), best_S, 0.0
