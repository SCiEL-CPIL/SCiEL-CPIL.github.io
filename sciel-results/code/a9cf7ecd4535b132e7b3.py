import numpy as np
import time
from scipy.spatial.distance import cdist

N = 16
D = 2

def get_hexagonal_points(n):
    # Generate hexagonal lattice points
    points = []
    # Radius large enough to get n points
    R = 10
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            # Hexagonal coordinates to Cartesian
            x = i + 0.5 * j
            y = (np.sqrt(3) / 2) * j
            points.append((x, y))
    points = np.array(points)
    # Sort by distance to origin
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]]

def normalize(points):
    """Center points at origin and scale so max pairwise distance is 1."""
    # Center
    points = points - np.mean(points, axis=0)
    # Calculate distances
    d = cdist(points, points)
    d_max = d.max()
    if d_max < 1e-10:
        return points
    points = points / d_max
    return points

def score(points):
    """Calculate (d_max / d_min)^2. Since we normalize d_max to 1, this is 1/d_min^2."""
    d = cdist(points, points)
    # Exclude diagonal
    np.fill_diagonal(d, np.inf)
    d_min = d.min()
    if d_min < 1e-10:
        return 1e10
    return 1.0 / (d_min * d_min)

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    best_points = None
    best_score = 1e10
    
    # Initial configurations
    inits = []
    
    # 1. Hexagonal
    hex_pts = get_hexagonal_points(N)
    hex_pts = normalize(hex_pts)
    inits.append(hex_pts)
    
    # 2. Random
    np.random.seed(42)
    rand_pts = np.random.randn(N, D)
    rand_pts = normalize(rand_pts)
    inits.append(rand_pts)
    
    # 3. Square grid
    grid_pts = np.array(list(__import__('itertools').product(range(4), repeat=2)), dtype=float)
    grid_pts = normalize(grid_pts)
    inits.append(grid_pts)
    
    # 4. Circle
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    circ_pts = np.column_stack([np.cos(theta), np.sin(theta)])
    circ_pts = normalize(circ_pts)
    inits.append(circ_pts)
    
    for init in inits:
        if time.time() - start_time > time_limit:
            break
            
        current_points = init.copy()
        current_score = score(current_points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = current_points.copy()
            
        # Local search
        no_improve_count = 0
        step_size = 0.1
        
        while time.time() - start_time < time_limit:
            # Pick a random point
            idx = np.random.randint(N)
            # Perturb
            noise = np.random.randn(D) * step_size
            new_points = current_points.copy()
            new_points[idx] += noise
            
            # Normalize
            new_points = normalize(new_points)
            
            # Check validity (distinct points)
            d = cdist(new_points, new_points)
            np.fill_diagonal(d, np.inf)
            if d.min() < 1e-9:
                continue
                
            new_score = score(new_points)
            
            if new_score < current_score:
                current_points = new_points
                current_score = new_score
                no_improve_count = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                no_improve_count += 1
                # Adaptive step size
                if no_improve_count > 100:
                    step_size *= 0.8
                    no_improve_count = 0
                if step_size < 1e-6:
                    break
                    
    if best_points is None:
        # Fallback to grid
        side = int(np.ceil(N ** (1 / D)))
        grid = list(__import__('itertools').product(range(side), repeat=D))[:N]
        best_points = np.array(grid, dtype=float)
        
    return best_points
