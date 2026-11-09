import numpy as np
import math

N = 12

def axial_to_cartesian(q, r):
    x = math.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return x, y

def min_side_for_points(points, theta):
    max_proj = 0.0
    for k in range(6):
        angle = theta + k * math.pi / 3.0
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        for x, y in points:
            proj = x * cos_a + y * sin_a
            if proj > max_proj:
                max_proj = proj
    return 2.0 * max_proj / math.sqrt(3.0)

def optimize_theta(points):
    best_s = float('inf')
    best_theta = 0.0
    num_samples = 2000
    for i in range(num_samples):
        theta = i * (math.pi / 3.0) / num_samples
        s = min_side_for_points(points, theta)
        if s < best_s:
            best_s = s
            best_theta = theta
    return best_s, best_theta

def construct():
    candidates = []
    
    # 1. 3x4 Rectangle
    pts = []
    for q in range(3):
        for r in range(4):
            pts.append(axial_to_cartesian(q, r))
    candidates.append(pts)
    
    # 2. 2x6 Rectangle
    pts = []
    for q in range(2):
        for r in range(6):
            pts.append(axial_to_cartesian(q, r))
    candidates.append(pts)
    
    # 3. 7 + 5 Side-Midpoints (Consecutive)
    pts = [axial_to_cartesian(0, 0)]
    ring1 = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    for q,r in ring1:
        pts.append(axial_to_cartesian(q, r))
    ring2_mids = [(2,0), (2,-2), (0,-2), (-2,0), (-2,2), (0,2)]
    for i in range(5):
        pts.append(axial_to_cartesian(ring2_mids[i][0], ring2_mids[i][1]))
    candidates.append(pts)
    
    # 4. 7 + 5 Side-Midpoints (Spread: 0, 1, 3, 4, 5)
    pts = [axial_to_cartesian(0, 0)]
    for q,r in ring1:
        pts.append(axial_to_cartesian(q, r))
    for i in [0, 1, 3, 4, 5]:
        pts.append(axial_to_cartesian(ring2_mids[i][0], ring2_mids[i][1]))
    candidates.append(pts)
    
    best_score = float('inf')
    best_inner = []
    best_center = (0.0, 0.0)
    best_side = 0.0
    best_angle = 0.0
    
    for pts in candidates:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        centered_pts = [(x - cx, y - cy) for x, y in pts]
        
        s, theta = optimize_theta(centered_pts)
        
        if s < best_score:
            best_score = s
            best_side = s
            best_angle = math.degrees(theta)
            best_inner = [[x, y, 0.0] for x, y in centered_pts]
            best_center = (0.0, 0.0)
            
    return best_inner, best_center, best_side, best_angle
