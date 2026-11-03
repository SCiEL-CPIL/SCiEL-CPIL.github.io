"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Uses Simulated Annealing to minimize a repulsive energy function.
    The energy is the sum of inverse 4th power distances between all pairs of circles
    and between each circle and the square boundaries.
    This drives the circles to a configuration that maximizes the minimum distance.
    """
    rng = np.random.RandomState(42)
    
    # Initial guess: Jittered grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    points = grid[:N].astype(float)
    # Add jitter to break symmetry
    points += rng.uniform(-0.02, 0.02, size=points.shape)
    # Ensure within bounds initially
    points = np.clip(points, -0.49, 0.49)
    
    # Energy function: Sum of 1/d^4 for pairs and boundaries
    def energy(pts):
        # Pairwise
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        # Avoid division by zero
        np.fill_diagonal(dists, 1.0)
        # Only upper triangle to avoid double counting
        i_upper = np.triu_indices(N, k=1)
        pair_energy = np.sum(1.0 / (dists[i_upper] ** 4))
        
        # Boundary distances
        # Distance to each wall
        d_left = pts[:, 0] + 0.5
        d_right = 0.5 - pts[:, 0]
        d_bottom = pts[:, 1] + 0.5
        d_top = 0.5 - pts[:, 1]
        
        # Clip small negative values to avoid inf/nan if slightly out
        d_left = np.clip(d_left, 1e-6, None)
        d_right = np.clip(d_right, 1e-6, None)
        d_bottom = np.clip(d_bottom, 1e-6, None)
        d_top = np.clip(d_top, 1e-6, None)
        
        bound_energy = np.sum(1.0 / (d_left ** 4)) + np.sum(1.0 / (d_right ** 4)) + \
                       np.sum(1.0 / (d_bottom ** 4)) + np.sum(1.0 / (d_top ** 4))
        
        return pair_energy + 10.0 * bound_energy # Weight boundaries higher to keep them in
    
    # Simulated Annealing
    T = 0.1
    T_min = 1e-6
    alpha = 0.999
    n_iter = 20000
    step_size = 0.05
    
    current_energy = energy(points)
    best_points = points.copy()
    best_energy = current_energy
    
    for i in range(n_iter):
        # Generate neighbor
        idx = rng.randint(N)
        delta = rng.normal(0, step_size, size=2)
        new_points = points.copy()
        new_points[idx] += delta
        
        # Keep within bounds strictly for validity, or let energy handle it?
        # Let energy handle it, but clip to avoid huge penalties/NaNs
        new_points = np.clip(new_points, -0.5, 0.5)
        
        new_energy = energy(new_points)
        
        # Accept if better, or with probability exp(-(E_new - E_old)/T)
        if new_energy < current_energy:
            points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
        else:
            if rng.random() < np.exp(-(new_energy - current_energy) / T):
                points = new_points
                current_energy = new_energy
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    return best_points
# EVOLVE-BLOCK-END
