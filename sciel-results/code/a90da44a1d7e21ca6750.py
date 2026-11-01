import numpy as np
import time
import math

N = 50

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # 1. Initialization: Hexagonal lattice
    # Approximate optimal spacing for 50 circles in unit disk
    # Area of disk = pi. Area of 50 circles = 50 * pi * r^2.
    # pi = 50 * pi * r^2 => r = 1/sqrt(50) ~ 0.141.
    # But packing efficiency is < 1, so r is smaller. Known best ~ 0.1258.
    # Spacing s ~ 2*r ~ 0.25.
    
    def hex_lattice_init():
        points = []
        # Generate a grid of hexagonal points
        # Range of x, y
        # We want to cover the unit disk
        s = 0.26 # Initial spacing
        k = int(1.0 / s) + 2
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        # If we have too many, pick N closest to center? Or random N?
        # Better: Jitter and then optimize.
        # We need exactly N points.
        if len(points) < N:
             # Fallback to random
             return np.random.rand(N, 2) * 2 - 1
             
        # Select N points. To make it symmetric-ish, pick those closest to center?
        # Or just take the first N from a shuffled list?
        # Let's pick N points that are well distributed.
        # Sort by distance from center and take N? That might cluster too much in center.
        # Let's just take a random subset of N from the lattice points within the disk.
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]

    # 2. Objective Function
    def get_radius(pts):
        # pts: (N, 2)
        # Check if any point is outside
        dists_to_center = np.linalg.norm(pts, axis=1)
        min_boundary = np.min(1.0 - dists_to_center)
        
        # Pairwise distances
        # Using broadcasting
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        return min(min_pair / 2.0, min_boundary)

    # 3. Simulated Annealing
    points = hex_lattice_init()
    # Add small jitter to break symmetry
    points += np.random.normal(0, 0.01, (N, 2))
    # Ensure they are inside
    norms = np.linalg.norm(points, axis=1)
    for i in range(N):
        if norms[i] > 1.0:
            points[i] = points[i] / norms[i] * 0.99

    current_r = get_radius(points)
    best_r = current_r
    best_points = points.copy()

    T = 0.05  # Initial temperature
    T_min = 1e-5
    alpha = 0.999  # Cooling factor
    step_size = 0.05  # Initial step size

    while T > T_min:
        if time.time() - start_time > time_limit:
            break
            
        # Generate move
        i = np.random.randint(N)
        old_point = points[i].copy()
        # Random move
        move = np.random.normal(0, step_size, 2)
        new_point = old_point + move
        
        # Check if new point is inside
        if np.linalg.norm(new_point) > 1.0:
            # Reject or reflect? Just reject for simplicity
            continue
            
        points[i] = new_point
        new_r = get_radius(points)
        
        delta = new_r - current_r
        if delta > 0 or np.random.random() < math.exp(delta / T):
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
        else:
            points[i] = old_point
            
        T *= alpha
        # Adapt step size
        step_size *= 0.999

    # 4. Final Polish
    # Use a simple local search: try small moves and accept if improvement.
    # Or use scipy.optimize if available and fast enough.
    # Let's do a coordinate descent / random hill climbing with very small steps.
    
    step_size = 0.001
    T = 1e-4
    for _ in range(20000): # Fixed number of steps for polish
        if time.time() - start_time > time_limit:
            break
        i = np.random.randint(N)
        old_point = best_points[i].copy()
        move = np.random.normal(0, step_size, 2)
        new_point = old_point + move
        
        if np.linalg.norm(new_point) > 1.0:
            continue
            
        best_points[i] = new_point
        new_r = get_radius(best_points)
        
        if new_r > best_r:
            best_r = new_r
        else:
            best_points[i] = old_point

    return best_points
