import numpy as np
import time
from scipy.spatial import Delaunay
import itertools

N = 50

def _true_radius(points):
    """
    Compute the true radius for a configuration of N points.
    points: (N, 2) array
    Returns: max radius r
    """
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Minimum pair distance (i < j)
    min_pair_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            if dists[i, j] < min_pair_dist:
                min_pair_dist = dists[i, j]
    
    # Distance to walls
    abs_x = np.abs(points[:, 0])
    abs_y = np.abs(points[:, 1])
    max_coord = np.maximum(abs_x, abs_y)
    dist_to_wall = 0.5 - max_coord
    min_wall_dist = np.min(dist_to_wall)
    
    # Radius is half the minimum of the two distances
    r = min(min_pair_dist, min_wall_dist) / 2.0
    return r

def _softmin_objective(points_flat, beta=100.0):
    """
    Differentiable surrogate for the negative radius.
    Uses softmin to approximate the minimum distance.
    """
    points = points_flat.reshape(N, 2)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to inf so it doesn't affect softmin
    np.fill_diagonal(dists, np.inf)
    
    # Softmin of pairwise distances
    # softmin(x) = -1/beta * log(sum(exp(-beta * x)))
    # This approximates min(x) as beta -> inf
    min_pair_soft = -np.log(np.sum(np.exp(-beta * dists))) / beta
    
    # Distance to walls
    abs_x = np.abs(points[:, 0])
    abs_y = np.abs(points[:, 1])
    max_coord = np.maximum(abs_x, abs_y)
    dist_to_wall = 0.5 - max_coord
    
    # Softmin of wall distances
    min_wall_soft = -np.log(np.sum(np.exp(-beta * dist_to_wall))) / beta
    
    # Combined softmin
    # We want min(min_pair, min_wall)
    # softmin(a, b) = -1/beta * log(exp(-beta*a) + exp(-beta*b))
    combined = -np.log(np.exp(-beta * min_pair_soft) + np.exp(-beta * min_wall_soft)) / beta
    
    # Radius is half the min distance
    return -combined / 2.0

def _gradient_polish(points, max_iter=500, tol=1e-10):
    """
    Gradient descent on the softmin objective to polish a local configuration.
    """
    beta = 100.0
    lr = 0.01
    
    points = points.copy()
    prev_val = _softmin_objective(points.flatten(), beta)
    
    for _ in range(max_iter):
        # Numerical gradient
        grad = np.zeros_like(points)
        eps = 1e-7
        for i in range(N):
            for d in range(2):
                orig = points[i, d]
                points[i, d] = orig + eps
                val_plus = _softmin_objective(points.flatten(), beta)
                points[i, d] = orig - eps
                val_minus = _softmin_objective(points.flatten(), beta)
                points[i, d] = orig
                grad[i, d] = (val_plus - val_minus) / (2 * eps)
        
        # Gradient step
        points -= lr * grad
        
        # Clip to stay inside square
        points = np.clip(points, -0.4999, 0.4999)
        
        # Check convergence
        val = _softmin_objective(points.flatten(), beta)
        if abs(val - prev_val) < tol:
            break
        prev_val = val
    
    return points

def _local_search(points, max_iter=200, temp=0.01):
    """
    Topology-aware local search using Delaunay triangulation.
    Identifies close pairs and pushes them apart.
    """
    points = points.copy()
    
    for _ in range(max_iter):
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(points)
        except:
            break
        
        # Get edges from triangulation
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a,b), max(a,b)))
        
        # For each edge, compute the distance and if it's small, push apart
        improved = False
        for a, b in edges:
            d = points[a] - points[b]
            dist = np.linalg.norm(d)
            if dist < 0.01:  # Threshold for "close"
                # Push apart along the connecting vector
                if dist > 1e-10:
                    direction = d / dist
                    push = 0.001 * (0.01 - dist)
                    points[a] += push * direction
                    points[b] -= push * direction
                    improved = True
        
        # Also check wall constraints
        for i in range(N):
            x, y = points[i]
            if x < -0.49:
                points[i, 0] = -0.49
            elif x > 0.49:
                points[i, 0] = 0.49
            if y < -0.49:
                points[i, 1] = -0.49
            elif y > 0.49:
                points[i, 1] = 0.49
        
        if not improved:
            break
    
    return points

def _hex_init(n_circles):
    """
    Generate a hexagonal lattice initialization.
    """
    # Approximate spacing
    s = np.sqrt(2.0 / (n_circles * np.sqrt(3.0)))
    
    # Generate grid points
    rows = 12
    cols = 12
    points = []
    for i in range(rows):
        for j in range(cols):
            x = j * s + (0.5 * s if i % 2 == 1 else 0)
            y = i * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
            
    points = np.array(points)
    
    # Center the grid
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    points[:, 0] -= cx
    points[:, 1] -= cy
    
    # Scale to fit in square roughly
    max_x = np.max(np.abs(points[:, 0]))
    max_y = np.max(np.abs(points[:, 1]))
    scale = 0.45 / max(max_x, max_y)
    points *= scale
    
    # Randomly select n_circles
    idx = np.random.choice(len(points), n_circles, replace=False)
    return points[idx]

def _square_grid_init(n_circles):
    m = int(np.ceil(np.sqrt(n_circles)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    np.random.shuffle(grid)
    return grid[:n_circles]

def _random_init(n_circles):
    return np.random.uniform(-0.45, 0.45, size=(n_circles, 2))

def _jittered_hex(n_circles):
    points = _hex_init(n_circles)
    points += np.random.normal(0, 0.005, points.shape)
    return np.clip(points, -0.49, 0.49)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s buffer for safety
    
    best_radius = -1
    best_points = None
    
    # List of initializers
    initializers = [_hex_init, _square_grid_init, _random_init, _jittered_hex]
    
    # We will run multiple optimizations
    num_restarts = 15
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Select initializer
        init_func = initializers[i % len(initializers)]
        
        # Add small random noise to break symmetry
        x0 = init_func(N)
        x0 += np.random.normal(0, 0.01, x0.shape)
        
        # Ensure points are inside
        x0 = np.clip(x0, -0.49, 0.49)
        
        # Phase 1: Local search
        points = _local_search(x0, max_iter=100)
        
        # Phase 2: Gradient polish
        points = _gradient_polish(points, max_iter=200)
        
        # Evaluate
        r = _true_radius(points)
        if r > best_radius:
            best_radius = r
            best_points = points.copy()
            
        # Optional: Second local search pass
        if time.time() < deadline - 20:
            points2 = _local_search(points, max_iter=50)
            points2 = _gradient_polish(points2, max_iter=100)
            r2 = _true_radius(points2)
            if r2 > best_radius:
                best_radius = r2
                best_points = points2.copy()
    
    # Final check
    if best_points is None:
        # Fallback to grid
        return _square_grid_init(N)
        
    # Ensure finite and within bounds
    best_points = np.clip(best_points, -0.5, 0.5)
    
    return best_points

# EVOLVE-BLOCK-START
# The code above is the implementation.
# I will place it inside the block.
# EVOLVE-BLOCK-END
