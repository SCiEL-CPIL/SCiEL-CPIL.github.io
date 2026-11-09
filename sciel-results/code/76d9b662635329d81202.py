"""Rewritten program: physics-based repulsion + simulated annealing for 50-circle packing."""
import numpy as np
import time

N = 50
DEADLINE = 120.0  # seconds; evaluator kills at 150s, keep margin


def _min_radius(centres):
    """Compute the largest common radius for a given set of centres."""
    # Wall constraints: distance to each side
    wall = np.min(np.minimum(0.5 - np.abs(centres), axis=1))
    # Pairwise distances
    # Use broadcasting to compute all pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, wall)


def construct():
    start_time = time.time()
    best_centres = None
    best_r = -np.inf

    # Initialize with a grid as a reasonable starting point
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centres = grid[:N].copy()

    # Also try a few other initializations
    inits = []
    inits.append(centres.copy())

    # Random initialization
    np.random.seed(42)
    for seed in range(5):
        np.random.seed(seed)
        rand_c = np.random.uniform(-0.45, 0.45, size=(N, 2))
        inits.append(rand_c)

    # Hexagonal-like pattern
    hex_c = np.zeros((N, 2))
    row = 0
    col = 0
    for i in range(N):
        x = (col + (0.5 if row % 2 else 0)) * 0.1 - 0.5 + 0.05
        y = row * 0.0866 - 0.4 + 0.04
        hex_c[i] = [x, y]
        col += 1
        if col >= 6:
            col = 0
            row += 1
    inits.append(hex_c)

    # Precompute pairwise indices for efficiency
    idx_i, idx_j = np.triu_indices(N, k=1)

    def _pairwise_dists(c):
        """Compute all pairwise distances efficiently."""
        diff = c[idx_i] - c[idx_j]
        return np.sqrt(np.sum(diff ** 2, axis=1))

    def _wall_dists(c):
        """Compute minimum distance to walls for each circle."""
        return np.minimum(0.5 - np.abs(c), axis=1).min()

    def _objective(c):
        """Compute the minimum radius (to maximize)."""
        pd = _pairwise_dists(c)
        min_pd = pd.min() / 2.0
        wd = _wall_dists(c)
        return min(min_pd, wd)

    # Simulated annealing / repulsion-based optimization
    for init_idx, init_c in enumerate(inits):
        if time.time() - start_time > DEADLINE - 5:
            break

        c = init_c.copy()
        # Evaluate initial
        r_init = _objective(c)
        if r_init > best_r:
            best_r = r_init
            best_centres = c.copy()

        # Repulsion-based optimization with cooling
        # Use a force-based approach: repel circles from each other and from walls
        temp = 0.05
        cooling = 0.9995
        step_size = 0.02
        n_iter = 50000

        for it in range(n_iter):
            if time.time() - start_time > DEADLINE - 3:
                break

            # Compute forces
            # Pairwise repulsion
            diff = c[idx_i] - c[idx_j]
            dists = np.sqrt(np.sum(diff ** 2, axis=1)) + 1e-12
            # Repulsive force: stronger when closer
            force_mag = np.where(dists < 0.2, 1.0 / (dists + 0.01), 0.0)
            forces = np.zeros((N, 2))
            # Scatter forces back
            np.add.at(forces, idx_i, -force_mag[:, np.newaxis] * diff / dists[:, np.newaxis])
            np.add.at(forces, idx_j, force_mag[:, np.newaxis] * diff / dists[:, np.newaxis])

            # Wall repulsion
            wall_dist = 0.5 - np.abs(c)
            wall_forces = np.zeros_like(c)
            # For each circle, push away from nearest wall
            for axis in range(2):
                # Left wall
                mask = wall_dist[:, axis] < 0.05
                wall_forces[mask, axis] += np.where(c[mask, axis] < 0, 1.0, 0.0) * (0.05 - wall_dist[mask, axis]) * 10
                # Right wall
                mask = wall_dist[:, axis] < 0.05
                wall_forces[mask, axis] += np.where(c[mask, axis] > 0, 1.0, 0.0) * (0.05 - wall_dist[mask, axis]) * 10

            forces += wall_forces

            # Update positions with noise
            noise = np.random.normal(0, step_size, size=(N, 2))
            c_new = c + forces * 0.01 + noise

            # Keep within bounds (soft constraint)
            c_new = np.clip(c_new, -0.5, 0.5)

            # Evaluate
            r_new = _objective(c_new)
            r_old = _objective(c)

            if r_new >= r_old or np.random.random() < np.exp((r_new - r_old) / max(temp, 1e-10)):
                c = c_new
                if r_new > best_r:
                    best_r = r_new
                    best_centres = c.copy()

            temp *= cooling
            step_size *= 0.9999

    # Final refinement: local optimization using scipy
    if best_centres is not None and time.time() - start_time < DEADLINE - 5:
        try:
            from scipy.optimize import minimize

            def _neg_radius(c_flat):
                c = c_flat.reshape(N, 2)
                return -_objective(c)

            res = minimize(
                _neg_radius,
                best_centres.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 2),
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res.success or res.fun < _neg_radius(best_centres.flatten()):
                r_final = -res.fun
                if r_final > best_r:
                    best_r = r_final
                    best_centres = res.x.reshape(N, 2)
        except Exception:
            pass

    if best_centres is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centres = grid[:N]

    return best_centres
