import numpy as np
import time
import itertools

N = 11

# Pre-compute indices for all triangles (i, j, k) with i < j < k
tri_indices = list(itertools.combinations(range(N), 3))
I = np.array([t[0] for t in tri_indices])
J = np.array([t[1] for t in tri_indices])
K = np.array([t[2] for t in tri_indices])

def min_area(points):
    """
    Compute the minimum area of any triangle formed by the points.
    points: (N, 2) array
    """
    # Vectors from i to j and i to k
    v1 = points[J] - points[I]
    v2 = points[K] - points[I]
    
    # Cross product z-component: x1*y2 - y1*x2
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def inside_triangle(p):
    """
    Check if point p is inside the equilateral triangle (0,0), (1,0), (1/2, sqrt(3)/2).
    """
    x, y = p
    if y < 0:
        return False
    if x < 0:
        return False
    # Line from (1,0) to (1/2, sqrt(3)/2):
    # slope = (sqrt(3)/2 - 0) / (1/2 - 1) = -sqrt(3)
    # y - 0 = -sqrt(3) * (x - 1)
    # y = -sqrt(3)x + sqrt(3)
    # sqrt(3)x + y <= sqrt(3)
    if np.sqrt(3) * x + y > np.sqrt(3) + 1e-9:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s buffer for Python overhead/exit
    
    # 1. Generate Initial Seeds
    seeds = []
    
    # Seed 1: Triangular Grid (10 points) + Center
    # Grid points for side 4 (0 to 3)
    # p(i, j) where i is row 0..3, j is 0..i
    # x = j/4 + i/8 ? No, let's map to (0,0)-(1,0)-(0.5, sqrt(3)/2)
    # Standard triangular grid vertices:
    # (0,0), (1,0), (0.5, sqrt(3)/2)
    # We want 10 points.
    # Let's use a uniform grid in barycentric coordinates?
    # Or just generate a grid of 4x4 triangular cells?
    # Points at (i/4 * 1 + j/4 * 0.5, j/4 * sqrt(3)/2) ?
    # Let's stick to a simple perturbation of a regular grid.
    
    # Regular grid of 10 points (side 4 triangle)
    pts_grid = []
    for i in range(4):
        for j in range(i + 1):
            # Map (i,j) to coordinates
            # x = j/4 + i/8 ?
            # Let's use barycentric-like mapping or linear interpolation.
            # Vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
            # Point P = (1-u-v)A + uB + vC
            # For a grid, we can use u, v such that u+v <= 1.
            # Let's just use a simple affine grid.
            # x = (j + i/2) / 4 ?
            # If i=0, j=0 -> x=0.
            # If i=3, j=3 -> x = (3 + 1.5)/4 = 1.125 (Out of bounds)
            
            # Better:
            # x = (j + i/2) / (4) is wrong for the slanted edge.
            # Let's use the property that the triangle is convex.
            # We can generate points on a triangular lattice.
            # e1 = (1/4, 0), e2 = (1/8, sqrt(3)/8)
            # P = i*e2 + j*(e1 - e2)? No.
            
            # Simplest:
            # x = (j + i/2) / 4 is for a right triangle? No.
            # Let's just use random points for seed 2, and a "hexagonal" seed.
            
            # Actually, let's just use a well-spaced random seed and rely on SA.
            # But a structured seed is better.
            
            # Let's try:
            # x = (j + i/2) / 4
            # y = i * sqrt(3) / 8
            # Check bounds:
            # i=3, j=3: x = 3.5/4 = 0.875, y = 3*1.732/8 = 0.649
            # Check line: sqrt(3)*0.875 + 0.649 = 1.515 + 0.649 = 2.16 > 1.732. Outside.
            
            # Correct mapping for triangular grid inside triangle:
            # Use barycentric coordinates (u, v, w) with u+v+w=1.
            # Points: (i/4, j/4, (4-i-j)/4) for i+j <= 4?
            # If i+j <= 3, we have 10 points (0,0), (1,0)..(3,0), (0,1)..(2,1), (0,2),(1,2), (0,3).
            # Count:
            # i=0: j=0..3 (4)
            # i=1: j=0..2 (3)
            # i=2: j=0..1 (2)
            # i=3: j=0 (1)
            # Total 10.
            # P = u*A + v*B + w*C where A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2).
            # P = v*B + w*C = (v + 0.5w, 0.5*sqrt(3)*w)
            # Here u=i/4, v=j/4, w=(4-i-j)/4.
            
            u = i / 4.0
            v = j / 4.0
            w = (4 - i - j) / 4.0
            x = v * 1.0 + w * 0.5
            y = w * (np.sqrt(3) / 2)
            pts_grid.append([x, y])
            
    pts_grid = np.array(pts_grid)
    
    # Add 11th point: Centroid
    centroid = [0.5, np.sqrt(3) / 6]
    seed1 = np.vstack([pts_grid, [centroid]])
    seeds.append(seed1)
    
    # Seed 2: Perturbed Grid
    rng = np.random.RandomState(42)
    seed2 = seed1 + rng.normal(0, 0.05, seed1.shape)
    # Project back inside? SA will handle rejection, but initial must be valid.
    # Simple projection: clip to bounds?
    # For now, just keep it if valid, otherwise regenerate.
    valid_seed2 = True
    for p in seed2:
        if not inside_triangle(p):
            valid_seed2 = False
            break
    if valid_seed2:
        seeds.append(seed2)
    
    # Seed 3: Random
    seed3 = rng.random((N, 2))
    # Map to triangle
    # u, v ~ U(0,1) such that u+v < 1
    # This is tricky with simple random.
    # Use rejection sampling for seed 3
    seed3_pts = []
    while len(seed3_pts) < N:
        u, v = rng.random(2)
        if u + v < 1:
            x = u + 0.5 * v
            y = np.sqrt(3) / 2 * v
            seed3_pts.append([x, y])
    seed3 = np.array(seed3_pts)
    seeds.append(seed3)

    best_score = -1
    best_points = None
    
    for seed in seeds:
        current_points = seed.copy()
        current_min = min_area(current_points)
        
        if current_min > best_score:
            best_score = current_min
            best_points = current_points.copy()
            
        # Simulated Annealing
        T = 0.01
        T_min = 1e-6
        alpha = 0.999
        steps_per_check = 1000
        
        # Adaptive T based on current min?
        # Area scale is ~ 0.01. So T should be comparable.
        
        for step in range(100000):
            if time.time() > deadline:
                break
                
            # Pick random point
            idx = np.random.randint(N)
            old_point = current_points[idx].copy()
            
            # Perturb
            # Step size should decrease with T
            step_size = 0.05 * (T / 0.01)
            new_point = old_point + np.random.normal(0, step_size, 2)
            
            # Check bounds
            if not inside_triangle(new_point):
                # Reject or project? Rejection is safer for SA dynamics
                continue
                
            current_points[idx] = new_point
            new_min = min_area(current_points)
            
            delta = new_min - current_min
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_min = new_min
                if current_min > best_score:
                    best_score = current_min
                    best_points = current_points.copy()
            else:
                current_points[idx] = old_point
                
            T *= alpha
            if T < T_min:
                T = T_min
                
            if step % steps_per_check == 0:
                if time.time() > deadline:
                    break

    return best_points

