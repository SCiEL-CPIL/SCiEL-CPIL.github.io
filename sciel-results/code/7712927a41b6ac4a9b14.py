import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # seconds, safety margin before 120s limit

def _score(centers):
    """Compute the maximum radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall distances: min distance from each center to the boundary
    # Cube is [-0.5, 0.5]^3
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_pair / 2.0, min_wall)

def _init_repulsive():
    """Initialize with random points and separate them via repulsion."""
    centers = np.random.rand(N, 3) - 0.5
    
    # Simple repulsion steps to avoid initial overlaps
    for _ in range(100):
        # Calculate distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find nearest neighbor for each point
        nearest_idx = np.argmin(dists, axis=1)
        
        # Move away from nearest neighbor
        for i in range(N):
            j = nearest_idx[i]
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
            else:
                vec = np.random.randn(3)
                vec = vec / np.linalg.norm(vec)
            
            # Move a small step
            centers[i] += 0.01 * vec
            
            # Keep inside cube
            centers[i] = np.clip(centers[i], -0.49, 0.49)
            
    return centers

def _simulated_annealing(centers, start_time, time_budget):
    """Simulated annealing to maximize the minimum clearance."""
    best_centers = centers.copy()
    best_score = _score(centers)
    current_centers = centers.copy()
    current_score = best_score
    
    T = 0.02  # Initial temperature
    cooling = 0.99
    step_size = 0.05
    
    while T > 1e-5 and (time.time() - start_time) < time_budget:
        # Perturb one random center
        idx = np.random.randint(N)
        
        new_centers = current_centers.copy()
        new_centers[idx] += np.random.randn(3) * step_size
        new_centers[idx] = np.clip(new_centers[idx], -0.49, 0.49)
        
        new_score = _score(new_centers)
        delta = new_score - current_score
        
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_centers = new_centers
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_centers = current_centers.copy()
        
        T *= cooling
        step_size = max(0.001, step_size * 0.999)
        
    return best_centers, best_score

def _local_refinement(centers, start_time, time_budget):
    """Local optimization using scipy minimize on negative score."""
    best_centers = centers.copy()
    best_score = _score(centers)
    
    def objective(x):
        c = x.reshape(N, 3)
        return -_score(c)
    
    # Use L-BFGS-B with a few restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if (time.time() - start_time) > time_budget:
            break
        
        if restart == 0:
            x0 = centers.ravel()
        else:
            # Add small random perturbation
            x0 = centers.ravel() + np.random.randn(N * 3) * 0.005
            x0 = x0.reshape(N, 3)
            x0 = np.clip(x0, -0.49, 0.49).ravel()
        
        try:
            # L-BFGS-B is efficient for box-constrained problems
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 1000, 'maxls': 50})
            new_centers = result.x.reshape(N, 3)
            new_centers = np.clip(new_centers, -0.49, 0.49)
            new_score = _score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
        except Exception:
            pass
    
    return best_centers, best_score

def construct():
    start_time = time.time()
    
    # Start with repulsive initialization
    centers = _init_repulsive()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Phase 1: Simulated Annealing
    time_budget_sa = 60.0
    centers, score = _simulated_annealing(centers, start_time, time_budget_sa)
    if score > best_score:
        best_score = score
        best_centers = centers.copy()
    
    # Phase 2: Local Refinement
    time_budget_lr = 50.0
    centers, score = _local_refinement(centers, start_time, time_budget_lr)
    if score > best_score:
        best_score = score
        best_centers = centers.copy()
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return best_centers
    
    # Phase 3: If time remains, try another random start
    if elapsed < DEADLINE - 15:
        centers2 = _init_repulsive()
        time_budget_sa2 = 10.0
        centers2, score2 = _simulated_annealing(centers2, start_time, time_budget_sa2)
        
        if score2 > best_score:
            best_score = score2
            best_centers = centers2.copy()
        
        if time.time() - start_time < DEADLINE - 5:
            centers2, score2 = _local_refinement(centers2, start_time, DEADLINE - 5)
            if score2 > best_score:
                best_score = score2
                best_centers = centers2.copy()
    
    return best_centers
