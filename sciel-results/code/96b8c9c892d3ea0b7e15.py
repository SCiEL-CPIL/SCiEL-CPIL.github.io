"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(points):
    """Compute the maximum feasible radius for a set of points."""
    if len(points) < N:
        return -1.0
    # Distance to boundary
    dist_boundary = 1.0 - np.sqrt(np.sum(points**2, axis=1))
    # Pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(points, 'euclidean')
    min_pair = np.min(d) / 2.0
    return min(np.min(dist_boundary), min_pair)

def _local_optimize(points, time_limit=5.0):
    """Local optimization using gradient-free methods."""
    best_points = points.copy()
    best_score = _score(best_points)
    start_time = time.time()
    
    # Use Nelder-Mead for local optimization
    # We'll optimize the negative score
    def neg_score(pts):
        return -_score(pts)
    
    # Try Nelder-Mead with multiple restarts from different perturbations
    nelder_options = {
        'maxiter': 200,
        'xatol': 1e-7,
        'fatol': 1e-9,
    }
    
    for restart in range(3):
        if time.time() - start_time > time_limit:
            break
        
        if restart == 0:
            x0 = best_points.flatten()
        else:
            # Perturb
            x0 = best_points.flatten() + np.random.normal(0, 0.001, size=N*2)
        
        # Ensure validity during optimization
        def objective(x):
            pts = x.reshape(-1, 2)
            # Penalty if points go outside or overlap
            dist_boundary = 1.0 - np.sqrt(np.sum(pts**2, axis=1))
            from scipy.spatial.distance import pdist
            d = pdist(pts, 'euclidean')
            min_pair = np.min(d) / 2.0
            r = min(np.min(dist_boundary), min_pair)
            if r < 0:
                return 1.0  # Large penalty
            return -r
        
        res = minimize(objective, x0, method='Nelder-Mead', options=nelder_options)
        
        if res.success or res.fun < -best_score:
            new_points = res.x.reshape(-1, 2)
            new_score = _score(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial hexagonal lattice
    spacing = 0.1
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
    
    # Local optimization
    points, score = _local_optimize(points, time_limit=10.0)
    
    # Try simulated annealing for global improvement
    if time.time() < deadline - 60:
        current_points = points.copy()
        current_score = score
        best_points = points.copy()
        best_score = score
        T = 0.01
        T_min = 0.0001
        cooling = 0.99
        
        sa_start = time.time()
        while time.time() - sa_start < 50:
            # Random perturbation
            idx = np.random.randint(N)
            perturbation = np.random.normal(0, T, size=2)
            new_points = current_points.copy()
            new_points[idx] += perturbation
            
            # Check if still inside unit circle
            if np.sqrt(np.sum(new_points[idx]**2)) > 1.0:
                continue
            
            new_score = _score(new_points)
            delta = new_score - current_score
            
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                current_points = new_points
                current_score = new_score
                if new_score > best_score:
                    best_points = new_points.copy()
                    best_score = new_score
            
            T *= cooling
            if T < T_min:
                break
        
        points = best_points
        score = best_score
    
    # Final local polish
    if time.time() < deadline - 10:
        points, score = _local_optimize(points, time_limit=20.0)
    
    return points
# EVOLVE-BLOCK-END
