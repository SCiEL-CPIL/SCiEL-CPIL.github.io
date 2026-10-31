"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p, v0, v1, v2):
    # Barycentric technique
    v0x, v0y = v0
    v1x, v1y = v1
    v2x, v2y = v2
    px, py = p
    
    d0 = (px - v1x) * (v0y - v1y) - (v0x - v1x) * (py - v1y)
    d1 = (px - v2x) * (v1y - v2y) - (v1x - v2x) * (py - v2y)
    d2 = (px - v0x) * (v2y - v0y) - (v2x - v0x) * (py - v0y)
    
    has_neg = (d0 < 0) or (d1 < 0) or (d2 < 0)
    has_pos = (d0 > 0) or (d1 > 0) or (d2 > 0)
    
    return not (has_neg and has_pos)

def triangle_area(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def min_area(points):
    min_a = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = triangle_area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def objective(x):
    points = x.reshape(-1, 2)
    # Check if all points are inside the triangle
    for p in points:
        if not point_in_triangle(p, V0, V1, V2):
            return 1000.0 # Large penalty
    areas = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                a = triangle_area(points[i], points[j], points[k])
                if a < 1e-9:
                    return 1000.0
                areas.append(a)
    return -min(areas)

def generate_hex_init():
    """Generate points based on a hexagonal lattice."""
    points = []
    # Spacing for hex grid. Approximate optimal spacing for 11 points in unit area tri?
    # Area of tri is sqrt(3)/4 ~ 0.433.
    # If we pack 11 points, average area per point is 0.433/11 ~ 0.039.
    # Side of equilateral tri per point ~ sqrt(0.039 * 4/sqrt(3)) ~ 0.3.
    # Let's try a few spacings.
    spacings = [0.25, 0.3, 0.35, 0.4]
    candidates = []
    for s in spacings:
        # Generate hex grid points
        h = s * np.sqrt(3) / 2
        # Iterate over a square region covering the triangle
        for i in range(-1, 10):
            for j in range(-1, 10):
                x = i * s + (j % 2) * s / 2
                y = j * h
                p = np.array([x, y])
                if point_in_triangle(p, V0, V1, V2):
                    candidates.append(p)
    
    if len(candidates) == 0:
        return None
    
    # If we have more than N candidates, pick N that are well distributed.
    # Simple strategy: Farthest Point Sampling
    if len(candidates) > N:
        # Start with a random point or center
        start_idx = np.random.randint(len(candidates))
        selected = [candidates[start_idx]]
        min_dists = np.array([np.linalg.norm(c - selected[0]) for c in candidates])
        
        for _ in range(N - 1):
            idx = np.argmax(min_dists)
            selected.append(candidates[idx])
            for i, c in enumerate(candidates):
                d = np.linalg.norm(c - selected[-1])
                if d < min_dists[i]:
                    min_dists[i] = d
        return np.array(selected)
    else:
        # Pad with random points if not enough
        pts = np.array(candidates)
        while len(pts) < N:
            u, v = np.random.random(), np.random.random()
            if u + v < 1:
                p = u * V1 + v * V2
                pts = np.vstack([pts, p])
            else:
                continue
        return pts[:N]

def generate_random_init():
    points = []
    while len(points) < N:
        u, v = np.random.random(), np.random.random()
        if u + v < 1:
            p = u * V1 + v * V2
            points.append(p)
    return np.array(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    best_score = -float('inf')
    best_points = None
    
    # Generate diverse initializations
    inits = []
    
    # 1. Hex based
    hex_init = generate_hex_init()
    if hex_init is not None:
        inits.append(hex_init)
    
    # 2. Random inits
    for _ in range(5):
        inits.append(generate_random_init())
        
    # 3. Grid based (distorted)
    # Create a 4x4 grid and map to triangle? 
    # Simple: Place points on a grid and jitter
    grid_pts = []
    for i in range(4):
        for j in range(4):
            u = (i + 0.5) / 4.0
            v = (j + 0.5) / 4.0
            if u + v < 1:
                p = u * V1 + v * V2
                grid_pts.append(p)
    if len(grid_pts) >= N:
        # Randomly pick N from grid
        indices = np.random.choice(len(grid_pts), N, replace=False)
        inits.append(np.array([grid_pts[i] for i in indices]))
        
    # Shuffle inits
    np.random.shuffle(inits)
    
    for init in inits:
        if time.time() > deadline:
            break
            
        x0 = init.flatten()
        try:
            res = minimize(objective, x0, method='Powell', 
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.success or res.fun < best_score:
                # Verify validity
                pts = res.x.reshape(-1, 2)
                valid = True
                for p in pts:
                    if not point_in_triangle(p, V0, V1, V2):
                        valid = False
                        break
                if valid:
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_points = pts
        except Exception:
            pass
            
    if best_points is None:
        # Fallback to random
        best_points = generate_random_init()
        
    return best_points

# EVOLVE-BLOCK-END
