import numpy as np
import math
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def axial_to_cartesian(coords):
        pts = []
        for q, r in coords:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        return np.array(pts)

    def center_cluster(pts):
        centroid = np.mean(pts, axis=0)
        return pts - centroid

    def min_container_side(pts, theta):
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rot_pts = pts @ R.T
        
        # Container hexagon at angle 0.
        # Normals at 30, 90, 150, 210, 270, 330 degrees.
        angles = [math.radians(30 + 60 * k) for k in range(6)]
        normals = np.array([[math.cos(a), math.sin(a)] for a in angles])
        
        projs = rot_pts @ normals.T
        max_proj = np.max(np.abs(projs))
        
        return 2.0 * max_proj / math.sqrt(3)

    def optimize_rotation(pts):
        thetas = np.linspace(0, math.pi/3, 100)
        best_theta = thetas[0]
        best_side = float('inf')
        
        for th in thetas:
            side = min_container_side(pts, th)
            if side < best_side:
                best_side = side
                best_theta = th
        
        # Refine
        if time.time() < deadline - 10:
            for _ in range(20):
                step = (math.pi/3 - 0) / 10000
                for th in np.linspace(max(0, best_theta - step), min(math.pi/3, best_theta + step), 100):
                    side = min_container_side(pts, th)
                    if side < best_side:
                        best_side = side
                        best_theta = th
        return best_side, best_theta

    candidates = []
    
    # C1: 3x3 + 2
    c1 = [(q, r) for q in range(3) for r in range(3)]
    c1 += [(3, 0), (3, 1)]
    candidates.append(c1)
    
    # C2: 7 + 4 (symmetric)
    c2 = [(0,0), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (0,1)]
    c2 += [(2,-1), (1,-2), (-1,-2), (-2,-1)]
    candidates.append(c2)
    
    # C3: 6+5
    c3 = [(q, 0) for q in range(6)] + [(q, 1) for q in range(5)]
    candidates.append(c3)
    
    # C4: 4+4+3
    c4 = [(q, 0) for q in range(4)] + [(q, 1) for q in range(4)] + [(q, 2) for q in range(3)]
    candidates.append(c4)
    
    # C5: 7 + 4 (different)
    c5 = [(0,0), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (0,1)]
    c5 += [(2,0), (2,-1), (-2,0), (-2,1)]
    candidates.append(c5)

    # C6: 1 center, 6 ring, 4 ring (all on one side?)
    c6 = [(0,0), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (0,1)]
    c6 += [(2,0), (2,-1), (2,-2), (1,-2)]
    candidates.append(c6)

    best_side = float('inf')
    best_theta = 0.0
    best_pts = None

    for coords in candidates:
        if time.time() > deadline - 5:
            break
        pts = axial_to_cartesian(coords)
        pts = center_cluster(pts)
        side, theta = optimize_rotation(pts)
        if side < best_side:
            best_side = side
            best_theta = theta
            c, s = math.cos(theta), math.sin(theta)
            R = np.array([[c, -s], [s, c]])
            best_pts = pts @ R.T

    if best_pts is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    angle_deg = math.degrees(best_theta)
    inner = []
    for p in best_pts:
        inner.append([p[0], p[1], angle_deg])
    
    # Outer angle is 0.0 to match the calculation frame
    return inner, (0.0, 0.0), best_side, 0.0
