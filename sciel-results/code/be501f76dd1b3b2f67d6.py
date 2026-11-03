"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 50
DEADLINE = time.time() + 130.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _score(centres):
    """Compute the packing radius for given centres."""
    # Wall constraints
    walls = 0.5 - np.abs(centres)
    wall_min = walls.min()
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_min = dists.min() / 2.0
    
    return min(wall_min, pair_min)

def _initial_grid():
    """Create a grid-like initial configuration."""
    m = int(np.ceil(np.sqrt(N)))
    # Use a slightly better initial layout
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _random_initial(scale=0.45):
    """Create a random initial configuration."""
    return np.random.uniform(-scale, scale, (N, 2))

def _optimize_from_start(start, max_iter=5000, tol=1e-12):
    """Optimize from a starting configuration using gradient-free method."""
    def objective(x):
        centres = x.reshape(-1, 2)
        return -_score(centres)
    
    result = minimize(
        objective,
        start.flatten(),
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
    )
    return result.x.reshape(-1, 2)

def _refine_with_l_bfgs(centres, max_iter=2000):
    """Further refine using L-BFGS-B with smooth approximation."""
    def smooth_objective(x):
        c = x.reshape(-1, 2)
        # Wall terms
        walls = 0.5 - np.abs(c)
        wall_min = walls.min()
        
        # Pairwise
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=-1))
        np.fill_diagonal(dists, np.inf)
        pair_min = dists.min() / 2.0
        
        r = min(wall_min, pair_min)
        return -r
    
    result = minimize(
        smooth_objective,
        centres.flatten(),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12}
    )
    return result.x.reshape(-1, 2)

def _local_search(centres, n_iter=2000, step_scale=0.01):
    """Simple local search with random perturbations."""
    best = centres.copy()
    best_score = _score(best)
    
    for _ in range(n_iter):
        if time.time() > DEADLINE:
            break
        # Random perturbation
        candidate = best + np.random.normal(0, step_scale, (N, 2))
        # Clamp to square
        candidate = np.clip(candidate, -0.49, 0.49)
        
        cand_score = _score(candidate)
        if cand_score > best_score:
            best = candidate
            best_score = cand_score
            step_scale *= 0.95
        else:
            step_scale *= 0.99
    
    return best, best_score

def construct():
    np.random.seed(42)
    
    best_centres = None
    best_score = -np.inf
    
    # Start with grid
    start = _initial_grid()
    s = _score(start)
    if s > best_score:
        best_score = s
        best_centres = start.copy()
    
    # Try optimization from grid
    if time.time() < DEADLINE - 30:
        opt = _optimize_from_start(start, max_iter=3000)
        s = _score(opt)
        if s > best_score:
            best_score = s
            best_centres = opt.copy()
    
    # Try several random starts
    for attempt in range(5):
        if time.time() > DEADLINE - 20:
            break
        rand_start = _random_initial()
        s = _score(rand_start)
        if s > best_score:
            best_score = s
            best_centres = rand_start.copy()
        
        # Optimize from random start
        if time.time() < DEADLINE - 15:
            opt = _optimize_from_start(rand_start, max_iter=2000)
            s = _score(opt)
            if s > best_score:
                best_score = s
                best_centres = opt.copy()
    
    # Local search refinement
    if time.time() < DEADLINE - 10 and best_centres is not None:
        refined, ref_score = _local_search(best_centres, n_iter=3000, step_scale=0.005)
        if ref_score > best_score:
            best_score = ref_score
            best_centres = refined.copy()
    
    # Final L-BFGS refinement
    if time.time() < DEADLINE - 5 and best_centres is not None:
        final = _refine_with_l_bfgs(best_centres, max_iter=1000)
        s = _score(final)
        if s > best_score:
            best_score = s
            best_centres = final.copy()
    
    # Ensure validity
    if best_centres is None:
        best_centres = _initial_grid()
    
    # Clamp to valid range
    best_centres = np.clip(best_centres, -0.499, 0.499)
    
    return best_centres
# EVOLVE-BLOCK-END
