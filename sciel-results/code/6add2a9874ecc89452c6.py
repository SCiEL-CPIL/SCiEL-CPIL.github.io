import numpy as np
from scipy.optimize import minimize
import time

N = 50
DIM = 3
# Total time limit for the entire construct function
TIME_LIMIT = 140.0
START_TIME = time.time()

def _pairwise_distances_sq(centers):
    """Compute squared pairwise distances between all pairs of centers."""
    # centers shape (N, 3)
    # diff shape (N, N, 3)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # dist_sq shape (N, N)
    dist_sq = np.sum(diff ** 2, axis=2)
    return dist_sq

def _wall_distances(centers):
    """Compute distance from each center to the nearest wall."""
    # centers shape (N, 3)
    # dist to each wall face
    # walls at -0.5 and 0.5
    # dist = 0.5 - abs(x)
    return 0.5 - np.abs(centers)

def _energy_and_grad(x):
    """
    Compute energy and gradient for the optimization.
    x: flat array of shape (3*N,)
    """
    centers = x.reshape(N, DIM)
    
    # 1. Pairwise Potential
    # We want to avoid division by zero. 
    # dist_sq will be 0 on diagonal.
    dist_sq = _pairwise_distances_sq(centers)
    # Set diagonal to 1 to avoid inf, we will mask it out
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy term: sum 1/d^2
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    d2_pairs = dist_sq[iu]
    
    # Avoid division by zero if points coincide (should not happen with random init)
    d2_pairs = np.maximum(d2_pairs, 1e-12)
    
    pair_energy = np.sum(1.0 / d2_pairs)
    
    # Gradient of pair energy
    # dE/dx_i = -2 * sum_{j != i} (x_i - x_j) / d_ij^4
    # We can compute this using the diff matrix
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # dist_sq is (N, N)
    # We need d_ij^4 = (d_ij^2)^2
    d4 = dist_sq ** 2
    np.fill_diagonal(d4, 1.0)
    
    # weight matrix W_ij = -2 / d_ij^4 for i != j, 0 for i == j
    W = -2.0 / d4
    np.fill_diagonal(W, 0.0)
    
    # grad_pair_i = sum_j W_ij * (x_i - x_j)
    # This is effectively W @ diff? No, diff is (N, N, 3).
    # grad_pair = np.sum(W[:, :, np.newaxis] * diff, axis=1)
    # W shape (N, N), diff shape (N, N, 3)
    # W[:, :, np.newaxis] shape (N, N, 1)
    # Broadcasting works.
    grad_pair = np.sum(W[:, :, np.newaxis] * diff, axis=1)
    
    # 2. Wall Potential
    # Energy term: sum_{i,k} 1 / (0.5 - |x_ik|)^2
    wall_dists = _wall_distances(centers)
    # Avoid division by zero if on wall
    wall_dists_safe = np.maximum(wall_dists, 1e-6)
    
    wall_energy = np.sum(1.0 / (wall_dists_safe ** 2))
    
    # Gradient of wall energy
    # dE/dx_ik = 2 * sign(x_ik) / (0.5 - |x_ik|)^3
    # sign(0) is 0, which is fine.
    signs = np.sign(centers)
    # If center is exactly 0, gradient is 0.
    # To be safe, if wall_dist is very small, force sign to 0? No, random init avoids this.
    grad_wall = 2.0 * signs / (wall_dists_safe ** 3)
    
    total_energy = pair_energy + wall_energy
    total_grad = grad_pair + grad_wall
    
    return total_energy, total_grad.flatten()

def _compute_score(centers):
    """Compute the actual packing score (max radius)."""
    # 1. Pairwise min distance
    dist_sq = _pairwise_distances_sq(centers)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # 2. Wall min distance
    wall_dists = _wall_distances(centers)
    min_wall_dist = np.min(wall_dists)
    
    # Radius is half of min pair dist, or min wall dist
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def construct():
    global START_TIME, TIME_LIMIT
    
    best_score = -1.0
    best_centers = None
    
    # Bounds for L-BFGS-B
    # Keep centers slightly inside to avoid boundary issues during optimization
    # The objective function pushes them away from walls anyway.
    bounds = [(-0.5, 0.5)] * (N * DIM)
    
    # Number of restarts
    # We will run until time limit is reached
    # Estimate time per restart. L-BFGS-B on 150 vars is fast, maybe 0.1-0.5s.
    # So we can do many restarts.
    
    rng = np.random.default_rng(42)
    
    restart_count = 0
    while True:
        elapsed = time.time() - START_TIME
        if elapsed > TIME_LIMIT - 5.0: # 5s margin
            break
        
        restart_count += 1
        
        # Initialization: Random points in a slightly smaller box
        # To avoid initial overlaps which cause huge gradients
        # A good init is a jittered grid or just random.
        # Let's try random uniform in [-0.4, 0.4]
        x0 = rng.uniform(-0.4, 0.4, size=(N, DIM)).flatten()
        
        try:
            res = minimize(
                _energy_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
            )
        except Exception as e:
            continue
            
        centers = res.x.reshape(N, DIM)
        
        # Check validity
        if not np.all(np.isfinite(centers)):
            continue
            
        score = _compute_score(centers)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            
    if best_centers is None:
        # Fallback: Simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    return best_centers
