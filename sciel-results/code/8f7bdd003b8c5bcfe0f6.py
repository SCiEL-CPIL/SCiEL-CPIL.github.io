"""Optimized program for 100 equal spheres in a unit cube using Simulated Annealing."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    np.random.seed(42)
    
    # Precompute pair indices for efficient distance computation
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def compute_radius(c):
        """Compute the maximum radius for given centers. c is (N, 3)."""
        min_wall = np.min(0.5 - np.abs(c))
        if min_wall <= 0:
            return 0.0
        diff = c[i_idx] - c[j_idx]
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        min_pair = np.min(dists)
        if min_pair <= 0:
            return 0.0
        r = min(min_pair / 2.0, min_wall)
        return max(r, 0.0)
    
    def energy(c):
        return -compute_radius(c)
    
    # Initial state: start from a grid and add small random perturbations
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    current = grid[:N].copy()
    current += np.random.normal(0, 0.005, size=current.shape)
    current = np.clip(current, -0.5, 0.5)
    
    current_energy = energy(current)
    best = current.copy()
    best_energy = current_energy
    
    # Annealing parameters
    T_start = 0.005
    T_end = 1e-7
    n_iter = 200000
    cooling = (T_end / T_start) ** (1.0 / n_iter)
    
    T = T_start
    
    for it in range(n_iter):
        idx = np.random.randint(0, N)
        new = current.copy()
        scale = 0.01 * (T / T_start + 0.1)
        new[idx] += np.random.normal(0, scale, size=3)
        new[idx] = np.clip(new[idx], -0.5, 0.5)
        
        new_energy = energy(new)
        delta = new_energy - current_energy
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current = new
            current_energy = new_energy
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy
        
        T *= cooling
    
    r = compute_radius(best)
    if r <= 0:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best
# EVOLVE-BLOCK-END
