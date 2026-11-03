"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def _score(radius, centres):
    """Compute the maximum common radius for given centres in the unit cube."""
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    
    # Compute pairwise distances
    diffs = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(centres)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def construct():
    """
    Use differential evolution to optimise sphere packing.
    We optimise the negative of the score to maximise it.
    """
    def objective(flat_centres):
        centres = flat_centres.reshape(N, 3)
        # Penalise out-of-bound centres heavily
        if np.any(centres < -0.5) or np.any(centres > 0.5):
            return 1.0  # Large penalty (we're minimising negative score)
        score = _score(None, centres)
        return -score
    
    # Initial population: random points in the cube
    np.random.seed(42)
    initial_pop = np.random.uniform(-0.48, 0.48, size=(50, N * 3))
    
    # Run differential evolution
    result = differential_evolution(
        objective,
        bounds=[(-0.49, 0.49)] * (N * 3),
        popsize=50,
        maxiter=100,
        tol=1e-8,
        seed=42,
        init='sobol',
        polish=False,
        update_interval=50
    )
    
    best_centres = result.x.reshape(N, 3)
    
    # Clamp to be strictly inside the cube
    best_centres = np.clip(best_centres, -0.4999, 0.4999)
    
    return best_centres
# EVOLVE-BLOCK-END
