"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def _compute_radius(centers):
        """Compute the maximum radius for the given centers."""
        if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
            return 0.0
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        if min_wall < 0:
            return 0.0
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        return min(min_pair, min_wall) / 2.0

    def _make_initial_grid():
        """Create a 5x5x2 grid as initial configuration."""
        nx, ny, nz = 5, 5, 2
        xs = -0.5 + (np.arange(nx) + 0.5) * (1.0 / nx)
        ys = -0.5 + (np.arange(ny) + 0.5) * (1.0 / ny)
        zs = -0.5 + (np.arange(nz) + 0.5) * (1.0 / nz)
        pts = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        return pts

    # --- PSO Setup ---
    n_particles = 40
    n_dims = N * 3  # 150
    w = 0.729       # inertia weight
    c1 = 1.49445    # cognitive
    c2 = 1.49445    # social
    max_iter = 3000

    # Initialize particles around the grid
    base_grid = _make_initial_grid().flatten()
    rng = np.random.default_rng(42)

    # Particle positions: (n_particles, n_dims)
    positions = np.zeros((n_particles, n_dims))
    velocities = np.zeros((n_particles, n_dims))
    pbest = np.zeros((n_particles, n_dims))
    pbest_val = np.full(n_particles, -np.inf)
    gbest = base_grid.copy()
    gbest_val = -np.inf

    for i in range(n_particles):
        # Perturb the grid
        noise_scale = 0.02
        pos = base_grid + rng.normal(0, noise_scale, n_dims)
        # Clip to valid range
        pos = np.clip(pos, -0.5, 0.5)
        positions[i] = pos
        v = rng.uniform(-0.01, 0.01, n_dims)
        velocities[i] = v
        centers = pos.reshape(N, 3)
        val = _compute_radius(centers)
        pbest[i] = pos
        pbest_val[i] = val
        if val > gbest_val:
            gbest_val = val
            gbest = pos.copy()

    # PSO loop
    for it in range(max_iter):
        if time.time() > deadline - 15:
            break

        r1 = rng.random(n_particles)
        r2 = rng.random(n_particles)

        for i in range(n_particles):
            # Update velocity
            velocities[i] = (w * velocities[i]
                             + c1 * r1[i] * (pbest[i] - positions[i])
                             + c2 * r2[i] * (gbest - positions[i]))
            # Update position
            positions[i] = positions[i] + velocities[i]
            # Clip to valid range
            positions[i] = np.clip(positions[i], -0.5, 0.5)
            # Evaluate
            centers = positions[i].reshape(N, 3)
            val = _compute_radius(centers)
            if val > pbest_val[i]:
                pbest_val[i] = val
                pbest[i] = positions[i].copy()
            if val > gbest_val:
                gbest_val = val
                gbest = positions[i].copy()

    # --- Nelder-Mead Polish ---
    # Start from the best PSO result
    best_centers = gbest.reshape(N, 3).copy()
    best_val = gbest_val

    def _nm_objective(x):
        c = x.reshape(N, 3)
        return -_compute_radius(c)

    try:
        res = minimize(_nm_objective, gbest, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < -best_val:
            best_centers = res.x.reshape(N, 3)
            best_val = -res.fun
    except Exception:
        pass

    # --- Final coordinate-wise local search ---
    # Try small perturbations of individual coordinates to squeeze out more
    if time.time() < deadline - 3:
        step = 0.001
        for _ in range(3):
            improved = False
            for i in range(N):
                for d in range(3):
                    for delta in [step, -step]:
                        trial = best_centers.copy()
                        trial[i, d] += delta
                        if np.all(trial >= -0.5) and np.all(trial <= 0.5):
                            val = _compute_radius(trial)
                            if val > best_val + 1e-12:
                                best_centers = trial
                                best_val = val
                                improved = True
            if not improved:
                break
            step *= 0.5

    # Final validation
    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers

# EVOLVE-BLOCK-END
