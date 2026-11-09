"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin before 150s kill

    # Start from cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centres = grid[:N].copy()
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    centres += rng.uniform(-0.001, 0.001, size=centres.shape)
    
    def score(x):
        """Compute the radius: min of half-pair-distances and wall-distances."""
        x = x.reshape(-1, 3)
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Wall distances
        wall = 0.5 - np.abs(x)
        min_wall = np.min(wall)
        
        return min(min_pair, min_wall)
    
    def soft_min(vals, beta=100.0):
        """Soft minimum via log-sum-exp."""
        m = np.min(vals)
        return m - np.log(np.sum(np.exp(beta * (vals - m)))) / beta
    
    def objective(x):
        """Smooth surrogate for the negative radius."""
        x = x.reshape(-1, 3)
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        wall = 0.5 - np.abs(x)
        
        # Combine all constraints
        all_vals = np.concatenate([dists[np.triu_indices_from(dists, k=1)] / 2.0, wall.ravel()])
        return -soft_min(all_vals, beta=50.0)
    
    # Local optimization with Nelder-Mead
    best_centres = centres.copy()
    best_score = score(best_centres)
    
    # Try multiple restarts with different perturbations
    for restart in range(3):
        if time.time() > deadline - 30:
            break
            
        # Perturb from best
        if restart > 0:
            centres = best_centres + rng.uniform(-0.01, 0.01, size=best_centres.shape)
        else:
            centres = best_centres + rng.uniform(-0.005, 0.005, size=best_centres.shape)
        
        result = minimize(objective, centres, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        if result.success or result.fun < objective(best_centres):
            new_centres = result.x.reshape(-1, 3)
            new_score = score(new_centres)
            if new_score > best_score:
                best_score = new_score
                best_centres = new_centres
        
        if time.time() > deadline - 20:
            break
    
    # Final polish pass with tighter tolerance
    if time.time() < deadline - 5:
        result = minimize(objective, best_centres, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if result.fun < objective(best_centres):
            polished = result.x.reshape(-1, 3)
            if score(polished) > best_score:
                best_centres = polished
    
    return best_centres
# EVOLVE-BLOCK-END

