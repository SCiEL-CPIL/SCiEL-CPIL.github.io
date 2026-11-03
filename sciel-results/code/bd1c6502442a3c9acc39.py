"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
import warnings

warnings.filterwarnings("ignore")


def _score(coords):
    """Compute the minimum radius for a given set of circle centers."""
    # coords shape (N, 2)
    # Wall constraint: min over i,k of (1/2 - |c_ik|)
    wall_dist = 0.5 - np.max(np.abs(coords), axis=1)
    min_wall = np.min(wall_dist)
    
    if min_wall < 0:
        return 0.0
    
    # Pairwise distances
    # Compute all pairwise distances efficiently
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    r = min(min_wall, min_pair / 2.0)
    return r


def _neg_score(coords):
    """Negative score for minimization."""
    return -_score(coords)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to leave margin before 150s kill
    
    # Initial guess: grid layout
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best_coords = grid[:N].copy()
    best_score = _score(best_coords)
    
    # Use differential evolution to maximize the minimum radius
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def objective(x):
        coords = x.reshape(-1, 2)
        return -_score(coords)
    
    # Try multiple random restarts with differential evolution
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        
        # Random initial population
        x0 = np.random.uniform(-0.5, 0.5, size=2 * N)
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                x0=x0,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                seed=restart,
                polish=True,
                init='sobol',
                updating='immediate'
            )
            
            if result.fun < -best_score:
                best_score = -result.fun
                best_coords = result.x.reshape(-1, 2)
                
        except Exception as e:
            pass
    
    # Additional local optimization using Nelder-Mead if time permits
    if time.time() < deadline - 10:
        from scipy.optimize import minimize
        
        try:
            res = minimize(
                _neg_score,
                best_coords.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res.fun < -best_score:
                best_score = -res.fun
                best_coords = res.x.reshape(-1, 2)
        except Exception:
            pass
    
    # Final check: ensure coordinates are within bounds
    best_coords = np.clip(best_coords, -0.5, 0.5)
    
    return best_coords
# EVOLVE-BLOCK-END
