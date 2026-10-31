import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

def get_score(points):
    """
    Calculate the squared ratio of max to min pairwise distance.
    points: (N, 2) array
    """
    # Compute pairwise squared distances
    # (N, 1, 2) - (1, N, 2) -> (N, N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    
    # Exclude diagonal
    np.fill_diagonal(dists_sq, np.inf)
    
    min_d2 = np.min(dists_sq)
    max_d2 = np.max(dists_sq)
    
    if min_d2 < 1e-12:
        return 1e10
        
    return max_d2 / min_d2

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = np.inf
    best_points = None
    
    # Generate initial candidates
    candidates = []
    
    # 1. 4x4 Grid
    grid = np.array(np.meshgrid(range(4), range(4))).T.reshape(-1, 2).astype(float)
    grid = (grid - grid.mean(axis=0)) / (grid.std(axis=0) + 1e-6)
    candidates.append(grid)
    
    # 2. Perturbed Grid
    candidates.append(grid + np.random.normal(0, 0.2, (N, 2)))
    
    # 3. Circle
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii = np.ones(N)
    circle = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    candidates.append(circle)
    
    # 4. Perturbed Circle
    candidates.append(circle + np.random.normal(0, 0.2, (N, 2)))
    
    # 5. Random Uniform
    for _ in range(50):
        candidates.append(np.random.uniform(-1, 1, (N, 2)))
    
    # Shuffle candidates to ensure diversity in early attempts
    np.random.shuffle(candidates)
    
    for init in candidates:
        if time.time() > deadline - 2.0:
            break
            
        x0 = init.flatten()
        
        try:
            # Use Powell method for robustness against non-smooth objective
            res = minimize(
                get_score, 
                x0, 
                method='Powell', 
                options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-6}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(N, 2)
                
        except Exception:
            continue
            
    if best_points is None:
        # Fallback to the grid if all optimizations failed
        best_points = grid
        
    return best_points
