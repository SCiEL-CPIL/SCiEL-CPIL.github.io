"""Program for packing 200 equal circles in a unit square using Simulated Annealing."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Initialize with a jittered grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, size=centers.shape)
    centers = np.clip(centers, -0.499, 0.499)

    # Energy function: Minimize sum of inverse squared distances and wall distances
    def energy(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)
        
        inv_dists_sq = 1.0 / (dists_sq + 1e-6)
        pair_energy = np.sum(inv_dists_sq)
        
        dx = 0.5 - np.abs(c[:, 0])
        dy = 0.5 - np.abs(c[:, 1])
        wall_dist = np.minimum(dx, dy)
        wall_dist_safe = np.maximum(wall_dist, 1e-6)
        boundary_energy = np.sum(1.0 / (wall_dist_safe ** 2))
        
        return pair_energy + boundary_energy

    current_centers = centers.copy()
    current_energy = energy(current_centers)
    
    # SA Parameters
    T = 1.0
    T_min = 1e-6
    cooling_rate = 0.9995
    max_iter = 50000
    num_particles = 5
    
    for i in range(max_iter):
        new_centers = current_centers.copy()
        idxs = rng.integers(0, N, size=num_particles)
        step_size = 0.01 * T
        moves = rng.normal(0, step_size, size=(num_particles, 2))
        new_centers[idxs] += moves
        
        new_energy = energy(new_centers)
        
        delta = new_energy - current_energy
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_centers = new_centers
            current_energy = new_energy
        
        T *= cooling_rate
        if T < T_min:
            break

    # Final clamp to ensure validity
    centers = np.clip(current_centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
