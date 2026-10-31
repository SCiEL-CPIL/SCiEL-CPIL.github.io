import numpy as np
import time
import itertools

N = 11
# Triangle vertices for unit area
# Area = sqrt(3)/4 * L^2 = 1 => L = sqrt(4/sqrt(3))
L = np.sqrt(4.0 / np.sqrt(3.0))
V1 = np.array([0.0, 0.0])
V2 = np.array([L, 0.0])
V3 = np.array([L/2.0, np.sqrt(3.0)/2.0 * L])

def barycentric_coords(p, v1, v2, v3):
    """Calculate barycentric coordinates of point p with respect to triangle v1, v2, v3."""
    # Using the determinant method
    # Total area
    denom = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1])
    
    # Area of sub-triangles
    # lambda1 corresponds to v1 (opposite to edge v2-v3)
    lam1 = ((v2[0] - p[0]) * (v3[1] - v2[1]) - (v3[0] - v2[0]) * (v2[1] - p[1])) / denom
    # lambda2 corresponds to v2
    lam2 = ((v3[0] - p[0]) * (v1[1] - v3[1]) - (v1[0] - v3[0]) * (v3[1] - p[1])) / denom
    # lambda3 corresponds to v3
    lam3 = 1.0 - lam1 - lam2
    
    return lam1, lam2, lam3

def is_inside_triangle(p, v1, v2, v3, epsilon=1e-9):
    l1, l2, l3 = barycentric_coords(p, v1, v2, v3)
    return l1 >= -epsilon and l2 >= -epsilon and l3 >= -epsilon

def project_to_triangle(p, v1, v2, v3):
    """Project point p to the triangle v1-v2-v3 if it is outside."""
    if is_inside_triangle(p, v1, v2, v3):
        return p
    
    # Find the closest point on the boundary
    # We can do this by clipping barycentric coordinates
    l1, l2, l3 = barycentric_coords(p, v1, v2, v3)
    
    # Clip to [0, 1]
    # This is a simplification; proper projection is more complex.
    # For optimization, a penalty might be better, or simple clamping.
    # Let's use a simple iterative projection or just penalty in the objective.
    # Given the speed, we can just use a penalty function.
    return p

def min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    # Using vector cross product for area: 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    # Vectorized calculation for all combinations is fast enough for N=11
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |det([p1-p3, p2-p3])|
                dx1 = p1[0] - p3[0]
                dy1 = p1[1] - p3[1]
                dx2 = p2[0] - p3[0]
                dy2 = p2[1] - p3[1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if area < min_area:
                    min_area = area
    return min_area

def penalty(points):
    """Penalty for points outside the triangle."""
    penalty_val = 0.0
    for p in points:
        if not is_inside_triangle(p, V1, V2, V3):
            # Calculate distance to triangle boundary roughly
            # Or just a large constant
            penalty_val += 10.0
    return penalty_val

def objective(points_flat):
    """Objective function for optimization. Returns negative min area (to maximize)."""
    points = points_flat.reshape(N, 2)
    
    # Check constraints
    outside_count = 0
    for p in points:
        if not is_inside_triangle(p, V1, V2, V3):
            outside_count += 1
            
    if outside_count > 0:
        # Heavy penalty
        return 1000.0 + outside_count
    
    area = min_triangle_area(points)
    return -area

def generate_initial_population(pop_size):
    """Generate diverse initial population."""
    pop = []
    
    # Strategy 1: Triangular Grid based
    # 4x4 grid has 15 points. We pick 11.
    # Grid points in barycentric coords (u, v) where u,v >= 0, u+v <= 1
    # Let's map grid indices (i, j) to barycentric
    # i, j from 0 to 3
    grid_points = []
    for i in range(4):
        for j in range(4 - i):
            # Barycentric: (1 - i/3 - j/3, i/3, j/3) ?
            # Let's use a simpler mapping
            u = i / 3.0
            v = j / 3.0
            if u + v <= 1.0:
                # Convert barycentric (1-u-v, u, v) to Cartesian
                # P = (1-u-v)V1 + uV2 + vV3
                p = (1.0 - u - v) * V1 + u * V2 + v * V3
                grid_points.append(p)
    
    # grid_points has 10 points (i=0..3, j=0..3-i)
    # Indices:
    # i=0: j=0,1,2,3 (4 pts)
    # i=1: j=0,1,2 (3 pts)
    # i=2: j=0,1 (2 pts)
    # i=3: j=0 (1 pt)
    # Total 10.
    # We need 11. Let's add the centroid.
    centroid = (V1 + V2 + V3) / 3.0
    grid_points.append(centroid)
    
    # Shuffle and take 11? They are already 11.
    # Add some noise
    for _ in range(pop_size // 3):
        pts = np.array(grid_points) + np.random.normal(0, 0.05, (N, 2))
        pop.append(pts.flatten())

    # Strategy 2: Hexagonal Cluster
    center = (V1 + V2 + V3) / 3.0
    radius = 0.2 * L
    hex_pts = []
    for k in range(6):
        angle = k * np.pi / 3
        p = center + radius * np.array([np.cos(angle), np.sin(angle)])
        hex_pts.append(p)
    hex_pts.append(center)
    # Add 4 more points at larger radius
    radius2 = 0.4 * L
    for k in range(4):
        angle = k * np.pi / 2 + np.pi/4
        p = center + radius2 * np.array([np.cos(angle), np.sin(angle)])
        hex_pts.append(p)
        
    for _ in range(pop_size // 3):
        pts = np.array(hex_pts) + np.random.normal(0, 0.05, (N, 2))
        pop.append(pts.flatten())

    # Strategy 3: Random
    for _ in range(pop_size - len(pop)):
        pts = np.random.rand(N, 2) * L
        # Simple rejection sampling for inside triangle
        valid_pts = []
        while len(valid_pts) < N:
            p = np.random.rand(2) * L
            if is_inside_triangle(p, V1, V2, V3):
                valid_pts.append(p)
        pop.append(np.array(valid_pts).flatten())
        
    return np.array(pop)

def differential_evolution(objective, pop_size=40, max_iter=100, F=0.8, CR=0.9, deadline=None):
    """Custom DE loop with deadline guard."""
    start_time = time.time()
    
    # Bounds: x in [0, L], y in [0, sqrt(3)/2 * L]
    bounds = [(0, L), (0, np.sqrt(3)/2 * L)] * N
    
    # Initialize
    population = generate_initial_population(pop_size)
    
    # Evaluate initial
    fitness = np.array([objective(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_individual = population[best_idx]
    best_fitness = fitness[best_idx]
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        for i in range(pop_size):
            # Select 3 distinct individuals
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Mutation
            mutant = population[r1] + F * (population[r2] - population[r3])
            
            # Crossover
            trial = np.copy(population[i])
            for j in range(len(mutant)):
                if np.random.random() < CR or j == np.random.randint(len(mutant)):
                    trial[j] = mutant[j]
            
            # Evaluate
            trial_fitness = objective(trial)
            
            # Selection
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_individual = trial.copy()
                    
    return best_individual

def construct():
    deadline = time.time() + 110.0 # 10s safety margin
    
    try:
        best_individual = differential_evolution(
            objective, 
            pop_size=50, 
            max_iter=200, 
            F=0.8, 
            CR=0.9, 
            deadline=deadline
        )
        
        points = best_individual.reshape(N, 2)
        
        # Final check: ensure points are inside
        # If any are outside, project them or return a fallback
        valid = True
        for p in points:
            if not is_inside_triangle(p, V1, V2, V3):
                valid = False
                break
        
        if not valid:
            # Fallback to a known good configuration (e.g., grid)
            # Generate a grid config
            grid_points = []
            for i in range(4):
                for j in range(4 - i):
                    u = i / 3.0
                    v = j / 3.0
                    p = (1.0 - u - v) * V1 + u * V2 + v * V3
                    grid_points.append(p)
            centroid = (V1 + V2 + V3) / 3.0
            grid_points.append(centroid)
            points = np.array(grid_points)
            
        return points

    except Exception as e:
        # Fallback
        grid_points = []
        for i in range(4):
            for j in range(4 - i):
                u = i / 3.0
                v = j / 3.0
                p = (1.0 - u - v) * V1 + u * V2 + v * V3
                grid_points.append(p)
        centroid = (V1 + V2 + V3) / 3.0
        grid_points.append(centroid)
        return np.array(grid_points)

# EVOLVE-BLOCK-END
