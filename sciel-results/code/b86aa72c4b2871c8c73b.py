"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    t0 = time.time()
    deadline = 130.0  # safety margin before 150s kill

    # Start with a simple grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    x0 = grid[:N].copy()

    def score(x):
        """Compute the radius for a given configuration."""
        x = x.reshape(N, 3)
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)

        # Distance to walls
        wall_dists = np.min(np.abs(0.5 - np.abs(x)), axis=1)
        min_wall_dist = np.min(wall_dists)

        return min(min_pair_dist / 2.0, min_wall_dist)

    def neg_score(x):
        """Negative score for minimization."""
        return -score(x)

    # Use L-BFGS-B to maximize the minimum distance
    # This is a non-smooth objective, but L-BFGS-B can still make progress
    best_x = x0.copy()
    best_score = score(best_x)

    # Try a few restarts with different initial configurations
    for restart in range(5):
        if time.time() > deadline:
            break

        if restart == 0:
            x_init = x0.copy()
        else:
            # Add small random perturbation
            rng = np.random.default_rng(restart * 12345 + 6789)
            x_init = x0 + rng.normal(0, 0.02, size=(N, 3))

        # Ensure initial configuration is valid
        x_init = np.clip(x_init, -0.5 + 0.01, 0.5 - 0.01)

        result = minimize(
            neg_score,
            x_init,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )

        if result.fun < -best_score:
            best_score = -result.fun
            best_x = result.x.reshape(N, 3)

        if time.time() > deadline:
            break

    # Final polish: verify and clip
    best_x = np.clip(best_x, -0.5, 0.5)

    # Ensure validity: no coincident centers, all inside cube
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < 1e-10:
                # Separate them slightly
                best_x[j] += 1e-5 * np.array([1, 0, 0])

    return best_x
# EVOLVE-BLOCK-END
