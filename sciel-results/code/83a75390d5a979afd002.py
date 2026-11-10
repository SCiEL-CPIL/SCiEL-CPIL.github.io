import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 50

def _relax(centers, steps=100, learning_rate=0.05):
    """
    Simple force-directed relaxation to spread points out.
    Uses repulsive forces between all pairs and wall repulsion.
    """
    for _ in range(steps):
        forces = np.zeros_like(centers)
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    # Random jitter if coincident
                    diff = np.random.normal(0, 1e-3, 3)
                    dist = np.linalg.norm(diff)
                # Force is proportional to 1/dist^2 (Coulomb-like)
                force_mag = 1.0 / (dist * dist)
                force_vec = (diff / dist) * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec
        
        # Wall repulsion
        for i in range(N):
            for k in range(3):
                # Distance to wall
                wall_dist = 0.5 - np.abs(centers[i, k])
                if wall_dist < 0.1: # Only repel if close to wall
                    # Force away from wall
                    sign = 1.0 if centers[i, k] > 0 else -1.0
                    forces[i, k] += sign * (1.0 / (wall_dist * wall_dist + 1e-6))

        # Update positions with simple gradient descent step
        centers += learning_rate * forces
        
        # Clip to bounds to prevent escaping
        centers = np.clip(centers, -0.49, 0.49)
        
    return centers

def _construct_initial(seed):
    """
    Generate a random initial configuration and relax it.
    """
    rng = np.random.default_rng(seed)
    # Start with random points
    centers = rng.uniform(-0.4, 0.4, (N, 3))
    
    # Relax to spread them out
    centers = _relax(centers, steps=50, learning_rate=0.1)
    
    return centers

def _objective(x_flat):
    """
    Objective function to minimize.
    Minimizes sum of inverse squared distances between pairs 
    and penalty for being too close to walls.
    """
    centers = x_flat.reshape(N, 3)
    
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 1e10
    
    # Pairwise distances
    dists = pdist(centers, metric='euclidean')
    # Avoid division by zero
    dists = np.maximum(dists, 1e-6)
    pair_energy = np.sum(1.0 / (dists ** 2))
    
    # Wall penalty
    # Distance to each wall
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    dz = 0.5 - np.abs(centers[:, 2])
    
    # If any distance is <= 0, huge penalty
    min_wall_dist = np.minimum(np.minimum(dx, dy), dz)
    if np.any(min_wall_dist <= 1e-6):
        return 1e10
    
    wall_energy = np.sum(1.0 / (min_wall_dist ** 2))
    
    return pair_energy + 100.0 * wall_energy

def _compute_score(centers):
    """Compute the minimum radius for the given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    dists = pdist(centers, metric='euclidean')
    min_pair = np.min(dists) / 2.0
    
    wall_dists = np.min(np.minimum(0.5 - np.abs(centers), 0.5 - np.abs(centers)), axis=1)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)

def construct():
    """Construct optimal sphere packing."""
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin
    
    best_score = 0.0
    best_centers = None
    
    # Define bounds
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Multi-start optimization
    num_restarts = 20  # Try up to 20 restarts
    
    for i in range(num_restarts):
        # Check time
        if time.time() - start_time > time_limit - 5:
            break
            
        seed = 42 + i
        initial_centers = _construct_initial(seed)
        
        x0 = initial_centers.flatten()
        
        # Calculate initial score
        init_score = _compute_score(initial_centers)
        
        # Run L-BFGS-B
        try:
            result = minimize(
                _objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6, 'maxfun': 5000}
            )
            
            opt_centers = result.x.reshape(N, 3)
            opt_score = _compute_score(opt_centers)
            
            if opt_score > best_score:
                best_score = opt_score
                best_centers = opt_centers.copy()
                
        except Exception as e:
            # If optimization fails, use the relaxed initial state if it's better
            if init_score > best_score:
                best_score = init_score
                best_centers = initial_centers.copy()

    # If we still have time, refine the best solution found
    if best_centers is not None and time.time() - start_time < time_limit - 10:
        try:
            # Use Nelder-Mead for finer tuning on the best candidate
            result2 = minimize(
                _objective, 
                best_centers.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'maxfev': 10000}
            )
            opt_centers2 = result2.x.reshape(N, 3)
            opt_centers2 = np.clip(opt_centers2, -0.5 + 1e-6, 0.5 - 1e-6)
            opt_score2 = _compute_score(opt_centers2)
            if opt_score2 > best_score:
                best_score = opt_score2
                best_centers = opt_centers2.copy()
        except:
            pass

    if best_centers is None:
        # Fallback: return a simple grid if everything failed
        centers = np.zeros((N, 3))
        for i in range(N):
            centers[i] = [0, 0, 0]
        return centers

    return best_centers
