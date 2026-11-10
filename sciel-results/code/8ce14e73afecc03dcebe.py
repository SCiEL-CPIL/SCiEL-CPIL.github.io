import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import random

def construct():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Initialization: Hexagonal packing
    # We want to place 50 circles.
    # A hexagonal lattice is denser than square.
    # Let's generate a grid of hexagonal points.
    
    def generate_hex_points(n_target):
        # Estimate spacing
        # Area per point in hex packing ~ (sqrt(3)/2) * s^2
        # Total area = 1
        # s ~ sqrt(1 / (n * sqrt(3)/2))
        s = np.sqrt(1.0 / (n_target * np.sqrt(3) / 2.0))
        
        points = []
        # Generate a range of points larger than the square to be safe
        range_x = np.arange(-0.7, 0.7, s / 2)
        range_y = np.arange(-0.7, 0.7, s / 2)
        
        for y in range_y:
            for x in range_x:
                # Hexagonal offset
                row_index = int(round(y / (s * np.sqrt(3) / 2)))
                if row_index % 2 == 0:
                    x_offset = 0
                else:
                    x_offset = s / 2
                px = x + x_offset
                py = y
                if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5:
                    points.append([px, py])
        
        points = np.array(points)
        
        # If we have too many, randomly select n_target
        if len(points) > n_target:
            indices = np.random.choice(len(points), n_target, replace=False)
            points = points[indices]
        elif len(points) < n_target:
            # If too few, add random points
            while len(points) < n_target:
                points = np.vstack([points, np.random.uniform(-0.5, 0.5, 2)])
                
        return points

    centers = generate_hex_points(N)
    
    # 2. Optimization: Maximize minimum distance
    # We define the objective as the negative of the minimum distance.
    # min_dist = min( min_{i<j} ||c_i - c_j||, min_i min_k (1 - 2*|c_ik|) )
    
    def objective(flat_centers):
        c = flat_centers.reshape(N, 2)
        
        # Calculate pairwise distances
        # Using broadcasting
        # dist[i, j] = ||c_i - c_j||
        # To save memory, we can compute in chunks or use a loop for N=50 (50*50 is small)
        d = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(c[i] - c[j])
                d[i, j] = dist
                d[j, i] = dist
        
        min_pair_dist = np.min(d[np.triu_indices(N, k=1)])
        
        # Calculate wall distances
        # dist to wall = 0.5 - |c_k|
        wall_dists = np.min(0.5 - np.abs(c), axis=1)
        min_wall_dist = np.min(wall_dists)
        
        min_dist = min(min_pair_dist, min_wall_dist)
        
        # We want to maximize min_dist, so we minimize -min_dist
        return -min_dist

    # Use Nelder-Mead or L-BFGS-B?
    # Nelder-Mead is derivative-free and handles non-smooth functions better.
    # With 100 variables, it might be slow, but N=50 is small enough.
    # Let's try L-BFGS-B with a smooth approximation if possible, or just Nelder-Mead.
    # Nelder-Mead can be slow. Let's use a simple gradient ascent with numerical gradients?
    # Or just use `minimize` with method='Nelder-Mead' and a reasonable maxiter.
    
    # Better: Use a local search with explicit gradient calculation for the active constraints.
    
    def local_search(centers, steps=5000, alpha=0.01):
        c = centers.copy()
        for step in range(steps):
            # Compute all distances
            d = np.zeros((N, N))
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.linalg.norm(c[i] - c[j])
                    d[i, j] = dist
                    d[j, i] = dist
            
            min_pair_dist = np.min(d[np.triu_indices(N, k=1)])
            
            wall_dists = np.min(0.5 - np.abs(c), axis=1)
            min_wall_dist = np.min(wall_dists)
            
            min_dist = min(min_pair_dist, min_wall_dist)
            
            # Find all pairs and walls that are within epsilon of min_dist
            eps = 1e-6
            grad = np.zeros_like(c)
            
            # Pair contributions
            for i in range(N):
                for j in range(i+1, N):
                    if d[i, j] <= min_dist + eps:
                        # Gradient of ||c_i - c_j|| wrt c_i is (c_i - c_j) / ||c_i - c_j||
                        # We want to increase distance, so move c_i away from c_j and c_j away from c_i
                        vec = c[i] - c[j]
                        norm = np.linalg.norm(vec)
                        if norm > 0:
                            grad[i] += vec / norm
                            grad[j] -= vec / norm
            
            # Wall contributions
            for i in range(N):
                for k in range(2):
                    # dist = 0.5 - |c_ik|
                    # If c_ik > 0, dist = 0.5 - c_ik, grad = -1
                    # If c_ik < 0, dist = 0.5 + c_ik, grad = +1
                    # We want to increase dist, so move c_ik away from 0.5 or -0.5
                    # i.e., towards 0
                    val = c[i, k]
                    if 0.5 - abs(val) <= min_dist + eps:
                        # Gradient of (0.5 - |val|) wrt val is -sign(val)
                        # To increase dist, we move in direction of grad
                        # grad_val = -sign(val)
                        # c_new = c + alpha * grad
                        # If val > 0, grad = -1, c decreases (moves to 0)
                        # If val < 0, grad = +1, c increases (moves to 0)
                        grad[i, k] += -np.sign(val) if val != 0 else 0.0

            # Normalize gradient to prevent large steps
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                c += alpha * (grad / grad_norm)
            
            # Clip to bounds
            c = np.clip(c, -0.5, 0.5)
            
            # Optional: Shrink alpha
            if step % 1000 == 0:
                alpha *= 0.9

        return c

    # Run local search
    centers = local_search(centers, steps=2000, alpha=0.05)
    
    # Final check and return
    return centers
# EVOLVE-BLOCK-END
