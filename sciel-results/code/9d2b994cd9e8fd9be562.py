import numpy as np
from scipy.optimize import minimize
import itertools
import time

N = 16
D = 2

def pairwise_dists_squared(points):
    # points: (N, 2)
    # returns (N, N) matrix of squared distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    return dists_sq

def soft_ratio(points, k=20.0):
    dists_sq = pairwise_dists_squared(points)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    d = dists_sq[mask]
    
    # Soft max and soft min
    # Use log-sum-exp for stability
    max_d = np.max(d)
    min_d = np.min(d)
    
    # Soft max: (1/k) * log(sum(exp(k * d)))
    # To avoid overflow, subtract max
    exp_kd = np.exp(k * (d - max_d))
    soft_max = max_d + (1.0/k) * np.log(np.sum(exp_kd))
    
    # Soft min: -(1/k) * log(sum(exp(-k * d)))
    # Subtract min for stability? No, subtract max of -d which is -min
    exp_neg_kd = np.exp(-k * (d - min_d)) # exp(-k(d-min)) = exp(k(min-d))
    soft_min = min_d - (1.0/k) * np.log(np.sum(exp_neg_kd))
    
    if soft_min < 1e-9:
        return 1e10
    
    return (soft_max / soft_min)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = float('inf')
    best_points = None
    
    # Initial configurations
    inits = []
    
    # 1. Grid
    side = int(np.ceil(N ** 0.5))
    grid = list(itertools.product(range(side), repeat=D))[:N]
    inits.append(np.array(grid, dtype=float))
    
    # 2. Hexagonal
    hex_points = []
    for row in range(4):
        for col in range(4):
            x = col + (row % 2) * 0.5
            y = row * np.sqrt(3)/2
            hex_points.append([x, y])
    inits.append(np.array(hex_points, dtype=float))
    
    # 3. Random
    np.random.seed(42)
    inits.append(np.random.uniform(-2, 2, (N, D)))
    
    k_values = [10.0, 20.0, 50.0]
    
    for k in k_values:
        for init in inits:
            if time.time() > deadline:
                break
            
            # Normalize initial points to avoid scaling issues
            # Center at 0
            init = init - np.mean(init, axis=0)
            
            def objective(x):
                pts = x.reshape(N, D)
                return soft_ratio(pts, k)
            
            # Try to optimize
            try:
                res = minimize(objective, init.flatten(), method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12})
                if res.success or res.fun < best_score:
                    pts = res.x.reshape(N, D)
                    # Calculate actual score
                    d_sq = pairwise_dists_squared(pts)
                    mask = ~np.eye(N, dtype=bool)
                    d = d_sq[mask]
                    actual_score = (np.max(d) / np.min(d))**2
                    if actual_score < best_score:
                        best_score = actual_score
                        best_points = pts
            except Exception:
                pass
            
            if time.time() > deadline:
                break

    if best_points is None:
        return np.array(inits[0], dtype=float)
    
    return best_points

# EVOLVE-BLOCK-END
