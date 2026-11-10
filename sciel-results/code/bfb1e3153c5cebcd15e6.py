import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Simulated Annealing approach
    # This is distinct from the lattice-based approach.
    
    rng = np.random.default_rng(42)
    
    # Initial configuration: Random uniform distribution
    # Or a jittered grid for better start? 
    # Let's use a jittered grid to give it a head start, which is still "from scratch" 
    # in the sense that we don't solve for lattice parameters.
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Add jitter
    jitter = rng.normal(0, 0.05, grid.shape)
    centers = grid[:N] + jitter
    # Clip to inside
    centers = np.clip(centers, -0.5 + 0.01, 0.5 - 0.01)

    # Potential Energy Function
    # We want to maximize the minimum distance.
    # Equivalent to minimizing a repulsive potential.
    # Use a high power to emphasize close pairs.
    # E = sum (1/d^p)
    # To handle boundaries, add sum (1/b^p)
    
    p_power = 8
    
    def calculate_energy(c):
        # c is (N, 3)
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0) # Avoid div by zero
        # Clip small distances to avoid inf
        dists = np.maximum(dists, 1e-9)
        pair_energy = np.sum(1.0 / np.power(dists, p_power))
        
        # Boundary distances
        # Distance to nearest face
        dx = 0.5 - np.abs(c[:, 0])
        dy = 0.5 - np.abs(c[:, 1])
        dz = 0.5 - np.abs(c[:, 2])
        b_dists = np.minimum(np.minimum(dx, dy), dz)
        b_dists = np.maximum(b_dists, 1e-9)
        bound_energy = np.sum(1.0 / np.power(b_dists, p_power))
        
        return pair_energy + bound_energy

    # Simulated Annealing Loop
    T_start = 10.0
    T_end = 1e-4
    n_steps = 20000
    # Geometric cooling
    alpha = (T_end / T_start) ** (1.0 / n_steps)
    
    current_energy = calculate_energy(centers)
    best_energy = current_energy
    best_centers = centers.copy()
    
    step_size = 0.05
    
    for step in range(n_steps):
        T = T_start * (alpha ** step)
        
        # Propose move: add Gaussian noise to all points
        noise = rng.normal(0, step_size, centers.shape)
        new_centers = centers + noise
        
        # Keep inside cube (hard constraint)
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_energy = calculate_energy(new_centers)
        
        # Metropolis criterion
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            centers = new_centers
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_centers = centers.copy()
        
        # Decay step size slightly
        if step % 1000 == 0 and step > 0:
            step_size *= 0.9

    # Final Polish: Local optimization to squeeze out a bit more
    # Use L-BFGS-B on the best center found by SA
    from scipy.optimize import minimize
    
    def objective(x):
        c = x.reshape(N, 3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0)
        dists = np.maximum(dists, 1e-9)
        pair_term = np.sum(1.0 / np.power(dists, p_power))
        
        dx = 0.5 - np.abs(c[:, 0])
        dy = 0.5 - np.abs(c[:, 1])
        dz = 0.5 - np.abs(c[:, 2])
        b_dists = np.minimum(np.minimum(dx, dy), dz)
        b_dists = np.maximum(b_dists, 1e-9)
        bound_term = np.sum(1.0 / np.power(b_dists, p_power))
        
        return pair_term + bound_term

    bounds = [(-0.5, 0.5)] * (N * 3)
    try:
        res = minimize(objective, best_centers.flatten(), method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
        if res.fun < best_energy:
            best_centers = res.x.reshape(N, 3)
    except Exception:
        pass

    return best_centers
# EVOLVE-BLOCK-END
