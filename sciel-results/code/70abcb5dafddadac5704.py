"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

# Seed for reproducibility
random.seed(42)
np.random.seed(42)


def _score(centres):
    """Compute the packing score (largest common radius)."""
    if centres.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(centres)):
        return -1.0
    # Check inside cube [-0.5, 0.5]^3
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return -1.0
    # Min distance to boundary
    dist_boundary = 0.5 - np.abs(centres)
    min_boundary = np.min(dist_boundary)
    if min_boundary < -1e-12:
        return -1.0
    # Pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(centres)
    min_pair = np.min(d)
    r = min(min_pair / 2.0, min_boundary)
    return r


def _objective(x, N_local):
    """Objective to minimize: negative of the packing score."""
    centres = x.reshape(N_local, 3)
    # Penalty for being outside the cube
    penalty = 0.0
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        penalty += 1.0
    # Compute pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(centres)
    min_pair = np.min(d)
    # Boundary distance
    dist_boundary = 0.5 - np.abs(centres)
    min_boundary = np.min(dist_boundary)
    r = min(min_pair / 2.0, min_boundary)
    return -r + penalty


def _random_init(N_local, rng):
    """Generate a random initial configuration."""
    # Use a jittered grid as starting point
    m = int(np.ceil(N_local ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N_local]
    # Add small random jitter
    jitter = rng.uniform(-0.01, 0.01, size=grid.shape)
    grid += jitter
    # Clip to cube
    grid = np.clip(grid, -0.49, 0.49)
    return grid


def _local_search(centres, rng, max_iter=5000):
    """Perform local optimization using BFGS."""
    x0 = centres.flatten()
    try:
        result = minimize(
            _objective,
            x0,
            args=(N,),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        if result.success or result.fun < 0:
            return result.x.reshape(N, 3)
        else:
            return centres
    except Exception:
        return centres


def _simulated_annealing(centres, rng, t_max=0.05, t_min=0.001, alpha=0.995, max_iter=3000):
    """Simulated annealing to escape local minima."""
    current = centres.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    t = t_max
    
    for i in range(max_iter):
        if t < t_min:
            break
        # Propose a move: perturb a random subset of points
        n_perturb = rng.randint(1, 5)
        indices = rng.choice(N, size=n_perturb, replace=False)
        proposed = current.copy()
        proposed[indices] += rng.normal(0, t * 0.5, size=(n_perturb, 3))
        proposed = np.clip(proposed, -0.49, 0.49)
        
        proposed_score = _score(proposed)
        delta = proposed_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / t):
            current = proposed
            current_score = proposed_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        
        t *= alpha
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    rng = np.random.RandomState(42)
    
    # Best solution found so far
    best_centres = None
    best_score = -1.0
    
    # Initial solution: jittered grid
    current = _random_init(N, rng)
    current_score = _score(current)
    if current_score > best_score:
        best_centres = current.copy()
        best_score = current_score
    
    # Run multiple restarts with local search and simulated annealing
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate new random initial configuration
        if restart == 0:
            init = current.copy()
        else:
            init = _random_init(N, rng)
        
        # Local search
        refined = _local_search(init, rng, max_iter=3000)
        refined_score = _score(refined)
        if refined_score > best_score:
            best_centres = refined.copy()
            best_score = refined_score
        
        if time.time() > deadline:
            break
        
        # Simulated annealing from the refined solution
        sa_result = _simulated_annealing(refined, rng, t_max=0.03, t_min=0.0005, alpha=0.997, max_iter=2000)
        sa_score = _score(sa_result)
        if sa_score > best_score:
            best_centres = sa_result.copy()
            best_score = sa_score
        
        # Final local polish
        if time.time() < deadline - 10:
            final = _local_search(best_centres, rng, max_iter=2000)
            final_score = _score(final)
            if final_score > best_score:
                best_centres = final.copy()
                best_score = final_score
    
    # Ensure validity
    if best_centres is None:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centres
# EVOLVE-BLOCK-END
