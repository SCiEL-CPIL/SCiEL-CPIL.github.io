"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _score(positions):
    """Compute the radius for a given set of positions.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    """
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(positions)
    min_wall = np.min(wall_dists)
    
    if min_wall <= 0:
        return 0.0
    
    # Pairwise distances
    # Using broadcasting is efficient for N=100
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)

def _jittered_grid_init(n, jitter=0.05):
    """Generate a jittered grid initialization."""
    # Find optimal grid size m such that m^3 >= n
    m = int(np.ceil(n ** (1.0/3.0)))
    # Create grid points
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # We have m^3 points, need n
    # Shuffle and take first n
    idx = np.random.permutation(len(grid))[:n]
    positions = grid[idx].copy()
    
    # Add jitter to break symmetry
    positions += np.random.uniform(-jitter, jitter, size=positions.shape)
    
    # Clamp to valid region
    positions = np.clip(positions, -0.5, 0.5)
    
    return positions

def _anneal(positions, start_time, time_limit, rng):
    """
    Metropolis-Hastings annealing.
    Returns best positions and score found during this run.
    """
    n = len(positions)
    current = positions.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    # Cooling parameters
    # Geometric cooling: T = T0 * alpha^iter
    # We want to finish around time_limit
    # Estimate iterations per second based on a quick test? 
    # Or just run until time runs out.
    
    t = 0.1
    t_end = 1e-5
    alpha = 0.999 # Cooling factor
    
    while time.time() < time_limit:
        # Step size proportional to temperature
        step = 0.01 + 0.05 * t
        
        # Choose move type: 
        # 80% chance: move 1 random sphere
        # 20% chance: move 2-3 random spheres
        if rng.random() < 0.8:
            n_move = 1
        else:
            n_move = rng.integers(2, 4)
        
        idx = rng.choice(n, size=n_move, replace=False)
        
        proposed = current.copy()
        delta = rng.normal(0, step, size=(n_move, 3))
        proposed[idx] += delta
        
        # Clamp
        proposed = np.clip(proposed, -0.5, 0.5)
        
        proposed_score = _score(proposed)
        
        # Acceptance
        if proposed_score >= current_score:
            current = proposed
            current_score = proposed_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            delta_e = proposed_score - current_score
            # Avoid underflow/overflow
            if delta_e / t > -50: 
                accept_prob = np.exp(delta_e / t)
            else:
                accept_prob = 0.0
            
            if rng.random() < accept_prob:
                current = proposed
                current_score = proposed_score
        
        t *= alpha
        if t < t_end:
            t = t_end
            
    return best, best_score

def construct():
    """Construct positions for 100 equal spheres in a unit cube."""
    n = N
    start_time = time.time()
    total_time_budget = 110.0 # Leave 10s buffer for safety
    
    best_positions = None
    best_score = 0.0
    
    # Phase 1: Multi-start with short runs
    # Try 5 different initializations
    for i in range(5):
        time_left = total_time_budget - (time.time() - start_time)
        if time_left < 5.0:
            break
            
        # Different jitter amounts
        jitter = 0.01 * (i + 1)
        init_pos = _jittered_grid_init(n, jitter=jitter)
        
        # Allocate time for this run
        run_time = min(10.0, time_left / (5 - i))
        run_end = time.time() + run_time
        
        rng = np.random.default_rng(42 + i)
        pos, score = _anneal(init_pos, start_time, run_end, rng)
        
        if score > best_score:
            best_score = score
            best_positions = pos
    
    # Phase 2: Refine the best found so far
    time_left = total_time_budget - (time.time() - start_time)
    if time_left > 10.0 and best_positions is not None:
        # Run a longer, finer annealing on the best candidate
        # Use a smaller step size and slower cooling
        rng = np.random.default_rng(123)
        
        # We can call _anneal again but with different parameters?
        # The current _anneal is hardcoded for alpha=0.999.
        # Let's make a specific refinement loop here for the best candidate.
        
        current = best_positions.copy()
        current_score = best_score
        best_local = current.copy()
        best_local_score = current_score
        
        t = 0.01
        t_end = 1e-6
        alpha = 0.9999 # Slower cooling
        
        while time.time() < start_time + total_time_budget - 2.0:
            step = 0.005 + 0.01 * t
            
            # Targeted move: move the sphere with min clearance?
            # Calculating min clearance sphere is O(N^2) extra.
            # For N=100, it's cheap.
            
            # Get distances to find "worst" sphere
            diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=-1))
            np.fill_diagonal(dists, np.inf)
            
            # Min distance for each sphere
            min_dists = np.min(dists, axis=1)
            # Also consider wall distance
            wall_dists = 0.5 - np.abs(current)
            min_walls = np.min(wall_dists, axis=1)
            
            # Effective clearance for each sphere
            clearances = np.minimum(min_dists / 2.0, min_walls)
            
            # Pick the sphere with the smallest clearance
            worst_idx = np.argmin(clearances)
            
            # Move this sphere
            proposed = current.copy()
            delta = rng.normal(0, step, size=3)
            proposed[worst_idx] += delta
            proposed = np.clip(proposed, -0.5, 0.5)
            
            proposed_score = _score(proposed)
            
            if proposed_score >= current_score:
                current = proposed
                current_score = proposed_score
                if current_score > best_local_score:
                    best_local = current.copy()
                    best_local_score = current_score
            else:
                delta_e = proposed_score - current_score
                if delta_e / t > -50:
                    accept_prob = np.exp(delta_e / t)
                else:
                    accept_prob = 0.0
                if rng.random() < accept_prob:
                    current = proposed
                    current_score = proposed_score
            
            t *= alpha
            if t < t_end:
                t = t_end
        
        if best_local_score > best_score:
            best_score = best_local_score
            best_positions = best_local
            
    return best_positions
# EVOLVE-BLOCK-END
