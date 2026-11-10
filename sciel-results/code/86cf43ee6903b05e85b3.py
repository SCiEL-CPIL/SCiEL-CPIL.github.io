"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(centres):
    """Compute the radius for a given set of centres."""
    n = len(centres)
    if n == 0:
        return 0.0
    # Wall constraint
    wall_dists = 0.5 - np.abs(centres)
    min_wall = np.min(wall_dists)
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_wall, min_pair)


def _optimize_repulsion(centres, time_limit=10.0):
    """Optimize centres using simulated annealing to maximize minimum distance."""
    start_time = time.time()
    best_centres = centres.copy()
    best_score = _compute_score(centres)
    current_score = best_score
    
    # Temperature schedule
    t0 = 0.01
    t_final = 1e-6
    cooling_rate = 0.999
    
    temp = t0
    n = len(centres)
    
    while temp > t_final and (time.time() - start_time) < time_limit:
        # Pick a random sphere and perturb it
        i = np.random.randint(n)
        delta = np.random.normal(0, temp, size=3)
        new_centres = centres.copy()
        new_centres[i] = new_centres[i] + delta
        
        # Check if new position is within bounds
        if np.any(np.abs(new_centres[i]) > 0.5):
            temp *= cooling_rate
            continue
            
        new_score = _compute_score(new_centres)
        
        # Accept if better or with probability exp((new_score - current_score) / temp)
        if new_score >= current_score or np.random.random() < np.exp((new_score - current_score) / temp):
            centres = new_centres
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_centres = centres.copy()
        
        temp *= cooling_rate
    
    return best_centres, best_score


def construct():
    """Construct 50 sphere centres in unit cube to maximize common radius."""
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Initial configuration: use a grid-like starting point for better convergence
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial = grid[:N].copy()
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    initial += np.random.normal(0, 0.001, size=initial.shape)
    
    best_centres = initial.copy()
    best_score = _compute_score(initial)
    
    # Multiple restarts with different random seeds
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() - start_time > deadline - 15.0:
            break
            
        # Different initial conditions for each restart
        if restart == 0:
            centres = initial.copy()
        else:
            centres = initial + np.random.normal(0, 0.01, size=initial.shape)
            # Clip to valid range
            centres = np.clip(centres, -0.49, 0.49)
        
        # Optimize with time limit per restart
        remaining_time = deadline - 15.0 - (time.time() - start_time)
        if remaining_time <= 1.0:
            break
            
        time_per_restart = min(remaining_time / (num_restarts - restart), 15.0)
        optimized, score = _optimize_repulsion(centres, time_limit=time_per_restart)
        
        if score > best_score:
            best_score = score
            best_centres = optimized.copy()
    
    # Final check: ensure all centres are within bounds
    best_centres = np.clip(best_centres, -0.499999, 0.499999)
    
    return best_centres
# EVOLVE-BLOCK-END
