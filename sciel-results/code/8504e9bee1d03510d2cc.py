import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Helper functions
    def hex_dist(x, y):
        # Distance from origin in hexagonal metric (flat-top outer hex)
        # L = max(|x|, 2/sqrt(3)*|y|, 2/sqrt(3)*|sqrt(3)/2*x - 1/2*y|, 2/sqrt(3)*|sqrt(3)/2*x + 1/2*y|)
        term1 = abs(x)
        term2 = (2.0 / math.sqrt(3.0)) * abs(y)
        term3 = (2.0 / math.sqrt(3.0)) * abs((math.sqrt(3.0) / 2.0) * x - 0.5 * y)
        term4 = (2.0 / math.sqrt(3.0)) * abs((math.sqrt(3.0) / 2.0) * x + 0.5 * y)
        return max(term1, term2, term3, term4)

    def get_vertices(cx, cy, theta):
        # Vertices of unit hexagon centered at (cx, cy) rotated by theta
        # Unit hexagon vertices (flat-top, side 1) are at angles 0, 60, 120, 180, 240, 300
        # Distance from center to vertex is 1.
        verts = []
        for k in range(6):
            angle = theta + k * math.pi / 3.0
            vx = cx + math.cos(angle)
            vy = cy + math.sin(angle)
            verts.append((vx, vy))
        return verts

    def check_overlap(params):
        # params: [x1, y1, t1, x2, y2, t2, ...]
        # Simple check: min distance between centers
        # For unit hexagons, min center distance is sqrt(3) if aligned.
        # If rotated, it can be smaller, but let's use sqrt(3) as a conservative constraint 
        # or a penalty.
        centers = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            centers.append((x, y))
        
        min_dist = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d = math.sqrt(dx * dx + dy * dy)
                if d < min_dist:
                    min_dist = d
        return min_dist

    def objective(params):
        # Calculate max hex_dist of all vertices
        max_d = 0.0
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            theta = params[i * 3 + 2]
            verts = get_vertices(x, y, theta)
            for vx, vy in verts:
                d = hex_dist(vx, vy)
                if d > max_d:
                    max_d = d
        
        # Overlap penalty
        min_dist = check_overlap(params)
        penalty = 0.0
        if min_dist < math.sqrt(3.0):
            # Strong penalty to push them apart
            penalty = (math.sqrt(3.0) - min_dist) * 10.0
            
        return max_d + penalty

    # Initial configuration: 3x4 block
    # Axial coords q in [0,3], r in [0,2]
    init_params = np.zeros(N * 3)
    idx = 0
    for q in range(4):
        for r in range(3):
            # Convert axial to cartesian
            x = math.sqrt(3.0) * (q + r / 2.0)
            y = 1.5 * r
            # Center the block
            # Block center approx: q=1.5, r=1 -> x=2*sqrt(3), y=1.5
            x_c = math.sqrt(3.0) * (1.5 + 0.5)
            y_c = 1.5 * 1.0
            init_params[idx * 3] = x - x_c
            init_params[idx * 3 + 1] = y - y_c
            init_params[idx * 3 + 2] = 0.0
            idx += 1

    best_solution = None
    best_score = float('inf')
    
    # Run optimization
    # Use Nelder-Mead
    options = {'maxiter': 1000, 'fatol': 1e-6}
    
    while time.time() < deadline - 10.0:
        # Add small noise to initial params for diversity
        noise = np.random.normal(0, 0.1, N * 3)
        # Keep rotations small initially
        for i in range(N):
            noise[i * 3 + 2] *= 0.1
        
        x0 = init_params + noise
        
        res = minimize(objective, x0, method='Nelder-Mead', options=options)
        
        if res.fun < best_score:
            best_score = res.fun
            best_solution = res.x
            
        # If we found a very good solution, break early?
        # L=2.5 is target. 
        if best_score < 2.6:
            break

    if best_solution is None:
        # Fallback to current best known
        # Use the 19-hex subset scaled to fit L=4.01
        centers = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                if abs(q + r) <= 2:
                    x = math.sqrt(3) * (q + r / 2.0)
                    y = 1.5 * r
                    centers.append((x, y))
        centers.sort(key=lambda p: p[0]**2 + p[1]**2)
        selected_centers = centers[:12]
        inner = []
        for x, y in selected_centers:
            inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 4.01, 0.0

    # Extract results
    inner = []
    for i in range(N):
        x = best_solution[i * 3]
        y = best_solution[i * 3 + 1]
        theta_rad = best_solution[i * 3 + 2]
        theta_deg = math.degrees(theta_rad)
        inner.append([x, y, theta_deg])
        
    outer_center = (0.0, 0.0)
    side = best_score
    angle = 0.0
    
    return inner, outer_center, side, angle

