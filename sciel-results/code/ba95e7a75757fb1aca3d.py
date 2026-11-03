"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.spatial.distance import cdist

def _score(centers):
    """Compute the minimum radius for a set of centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    # Compute pairwise distances
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    # Distance to walls
    min_wall = np.min(np.minimum(0.5 - np.abs(centers), 0.5 - np.abs(centers)))
    # Actually need min over all coordinates
    min_wall = np.min(np.minimum(np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
                                  np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1]),
                                  np.minimum(0.5 - centers[:, 2], 0.5 + centers[:, 2])))
    return min(min_pair, 2.0 * min_wall) / 2.0

def _initial_grid():
    """Create a simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _repulsion_energy(centers, k=20.0):
    """Compute repulsion energy and gradient."""
    d = cdist(centers, centers)
    np.fill_diagonal(d, 1.0)
    # Only consider pairs closer than some threshold
    mask = d < 0.5
    energy = 0.0
    grad = np.zeros_like(centers)
    for i in range(N):
        for j in range(i + 1, N):
            if mask[i, j]:
                dist = d[i, j]
                if dist > 1e-10:
                    force = k / (dist ** 3)
                    diff = centers[i] - centers[j]
                    grad[i] += force * diff
                    grad[j] -= force * diff
                    energy += k / (2.0 * dist ** 2)
    return energy, grad

def _wall_penalty(centers, strength=100.0):
    """Penalty for being outside the cube."""
    penalty = 0.0
    grad = np.zeros_like(centers)
    for i in range(N):
        for k in range(3):
            x = centers[i, k]
            if x < -0.5:
                penalty += strength * (-0.5 - x) ** 2
                grad[i, k] += 2.0 * strength * (-0.5 - x)
            elif x > 0.5:
                penalty += strength * (x - 0.5) ** 2
                grad[i, k] -= 2.0 * strength * (x - 0.5)
    return penalty, grad

def _local_optimize(centers, max_iter=500, lr=0.01):
    """Local optimization using gradient descent on repulsion + wall penalty."""
    centers = centers.copy()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    for it in range(max_iter):
        energy, grad_rep = _repulsion_energy(centers)
        penalty, grad_wall = _wall_penalty(centers)
        grad = grad_rep + grad_wall
        
        # Adaptive step size
        norm = np.linalg.norm(grad)
        if norm < 1e-10:
            break
        step = lr / (norm + 1e-6)
        step = min(step, 0.05)
        
        new_centers = centers - step * grad
        
        # Clip to cube
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_score = _score(new_centers)
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers.copy()
        
        if new_score >= _score(centers) - 1e-6:
            centers = new_centers
        else:
            # Try smaller step
            step *= 0.5
            new_centers = centers - step * grad
            new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
            centers = new_centers
    
    return best_centers

def _perturb(centers, sigma=0.05):
    """Randomly perturb centers."""
    perturbed = centers.copy()
    mask = np.random.random(N) < 0.3  # Perturb 30% of spheres
    indices = np.where(mask)[0]
    if len(indices) > 0:
        perturbations = np.random.normal(0, sigma, size=(len(indices), 3))
        perturbed[indices] += perturbations
        perturbed = np.clip(perturbed, -0.5 + 1e-9, 0.5 - 1e-9)
    return perturbed

def construct():
    """Construct the optimal sphere packing."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    # Start with grid initialization
    best_centers = _initial_grid()
    best_score = _score(best_centers)
    
    # Local optimization
    best_centers = _local_optimize(best_centers, max_iter=300)
    best_score = _score(best_centers)
    
    # Iterative perturbation and local optimization
    num_perturbations = 0
    while time.time() < deadline - 5.0:
        # Perturb the current best solution
        perturbed = _perturb(best_centers, sigma=0.03 + 0.02 * (1 - (time.time() - start_time) / 140.0))
        
        # Local optimization
        optimized = _local_optimize(perturbed, max_iter=200)
        score = _score(optimized)
        
        if score > best_score:
            best_score = score
            best_centers = optimized
            num_perturbations = 0
        else:
            num_perturbations += 1
            if num_perturbations > 10:
                # Reset perturbation count and try larger perturbations
                num_perturbations = 0
    
    return best_centers
# EVOLVE-BLOCK-END
