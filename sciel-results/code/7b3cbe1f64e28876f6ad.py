import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 14
D = 3

def _soft_max(x, tau):
    # Numerically stable soft max
    x_max = np.max(x)
    return x_max + tau * np.log(np.sum(np.exp((x - x_max) / tau)))

def _soft_min(x, tau):
    # Numerically stable soft min
    x_min = np.min(x)
    return x_min - tau * np.log(np.sum(np.exp(-(x - x_min) / tau)))

def _compute_distances(points):
    # points: (N, D)
    # Returns array of pairwise distances (N*(N-1)/2,)
    n = len(points)
    dists = np.zeros((n * (n - 1)) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            dists[idx] = np.linalg.norm(diff)
            idx += 1
    return dists

def _objective_and_grad(flat_points, tau=0.05):
    # flat_points: (36,) corresponding to points 2..13
    # Reconstruct points
    points = np.zeros((N, D))
    points[0] = [0, 0, 0]
    points[1] = [1, 0, 0]
    points[2:] = flat_points.reshape(N - 2, D)
    
    dists = _compute_distances(points)
    
    d_max = _soft_max(dists, tau)
    d_min = _soft_min(dists, tau)
    
    # Avoid division by zero
    if d_min < 1e-6:
        return 1e6, np.zeros_like(flat_points)
        
    ratio = d_max / d_min
    score = ratio * ratio
    
    # Gradient calculation via numerical differentiation or analytical?
    # L-BFGS-B can use numerical gradients if jac is not provided, 
    # but providing it is faster. 
    # However, analytical grad of soft max/min is complex.
    # Given small N, numerical gradient is acceptable if we run many iterations?
    # Actually, L-BFGS-B with numerical grad on 36 vars is quite fast.
    # Let's let scipy handle the gradient numerically first to ensure correctness.
    
    return score, None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_score = float('inf')
    best_points = None
    
    # Fixed points: P0=(0,0,0), P1=(1,0,0)
    # Variables: P2..P13 (12 points * 3 coords = 36 vars)
    
    # Try different temperatures for soft max/min
    taus = [0.01, 0.05, 0.1]
    
    # Number of restarts
    # We don't know how fast each run is, so we loop until deadline
    np.random.seed(42)
    
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        
        # Initialize remaining points
        # Randomly place in a box [-2, 2]
        rand_points = np.random.uniform(-2.0, 2.0, size=(N - 2, D))
        
        # Optimization
        x0 = rand_points.flatten()
        
        # Use a small tau for the main optimization to approximate min/max well
        # The tau in the objective function
        tau = taus[attempt % len(taus)]
        
        # We use L-BFGS-B
        # maxiter: 1000 should be enough for 36 vars
        # ftol: 1e-10
        try:
            res = minimize(
                _objective_and_grad, 
                x0, 
                args=(tau,), 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if res.success or res.fun < best_score:
                # Reconstruct full points to get the true score (not soft)
                points = np.zeros((N, D))
                points[0] = [0, 0, 0]
                points[1] = [1, 0, 0]
                points[2:] = res.x.reshape(N - 2, D)
                
                dists = _compute_distances(points)
                d_max = np.max(dists)
                d_min = np.min(dists)
                
                if d_min > 1e-6:
                    true_score = (d_max / d_min) ** 2
                    if true_score < best_score:
                        best_score = true_score
                        best_points = points.copy()
                        
        except Exception as e:
            # If optimization fails, continue
            pass
            
        # Check time
        if time.time() > deadline - 5.0:
            break

    # If no valid points found (should not happen), return seed-like result
    if best_points is None:
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
        
    return best_points

# EVOLVE-BLOCK-END
