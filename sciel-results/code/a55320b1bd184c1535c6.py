import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 14
D = 3

def _pairwise_sq_dists(points):
    # points: (N, D)
    # returns (N, N) matrix of squared distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    return dists_sq

def _objective(params):
    points = params.reshape(N, D)
    dists_sq = _pairwise_sq_dists(points)
    # Exclude diagonal
    mask = ~np.eye(N, dtype=bool)
    dists_sq = dists_sq[mask]
    if len(dists_sq) == 0:
        return 1e10
    min_d = np.min(dists_sq)
    max_d = np.max(dists_sq)
    if min_d < 1e-12:
        return 1e10
    return max_d / min_d

def _fibonacci_sphere(n, radius=1.0):
    # Generate n points on a sphere of given radius using Fibonacci lattice
    points = []
    for i in range(n):
        phi = np.arccos(1 - 2 * (i + 0.5) / n)
        theta = np.pi * (1 + 5**0.5) * i
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        points.append([x, y, z])
    return np.array(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_score = np.inf
    best_params = None
    
    # Initial guesses
    inits = []
    
    # 1. Fibonacci sphere
    fib_points = _fibonacci_sphere(N, radius=1.0)
    inits.append(fib_points.flatten())
    
    # 2. Random points in a cube
    for _ in range(5):
        rand_points = np.random.uniform(-1, 1, (N, D))
        # Center them
        rand_points -= np.mean(rand_points, axis=0)
        inits.append(rand_points.flatten())
        
    # 3. Points on a regular grid, perturbed
    side = int(np.ceil(N ** (1/D)))
    grid = list(itertools.product(range(side), repeat=D))[:N]
    grid_points = np.array(grid, dtype=float)
    grid_points -= np.mean(grid_points, axis=0)
    inits.append(grid_points.flatten())
    
    # Run optimization for each initial guess
    for i, x0 in enumerate(inits):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # If no optimization succeeded or didn't improve, fallback to best initial
    if best_params is None:
        # Evaluate all inits and pick the best
        for x0 in inits:
            score = _objective(x0)
            if score < best_score:
                best_score = score
                best_params = x0
                
    if best_params is None:
        # Absolute fallback
        return np.array(_fibonacci_sphere(N, radius=1.0))
        
    return best_params.reshape(N, D)
