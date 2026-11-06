"""Improved packing: grid-based initialization + simulated annealing with continuous moves."""
import numpy as np
import time

N = 50
DEADLINE = 115.0  # seconds, safety margin before 120s limit


def _score(positions):
    """Compute the radius for a given set of positions."""
    # Clip positions to be inside the cube
    positions = np.clip(positions, -0.5, 0.5)
    
    # Compute pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Minimum pairwise distance
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Minimum distance to walls
    wall_dists = 0.5 - np.abs(positions)
    min_wall_dist = np.min(wall_dists)
    
    # Radius is half the minimum of these
    r = min(min_pair_dist / 2.0, min_wall_dist)
    
    return r


def _score_fast(positions):
    """Faster score computation using vectorized operations."""
    positions = np.clip(positions, -0.5, 0.5)
    
    # Pairwise distances - use a more memory-efficient approach
    min_pair_dist = np.inf
    for i in range(N):
        diffs = positions[i+1:] - positions[i]
        d = np.sqrt(np.sum(diffs**2, axis=1))
        if len(d) > 0:
            min_pair_dist = min(min_pair_dist, np.min(d))
    
    wall_dists = 0.5 - np.abs(positions)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r


def _init_grid():
    """Initialize with a cubic grid arrangement."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    # We need m^3 >= N, so m = 4 gives 64 >= 50
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _init_random():
    """Initialize with random positions."""
    return np.random.uniform(-0.5, 0.5, size=(N, 3))


def _init_mixed():
    """Initialize with a mix of grid and random."""
    pos = _init_grid()
    # Add small random perturbations
    pos += np.random.normal(0, 0.01, size=pos.shape)
    return np.clip(pos, -0.5, 0.5)


def construct():
    start_time = time.time()
    
    # Try multiple initializations and keep the best
    best_pos = None
    best_score = -np.inf
    
    init_methods = [_init_grid, _init_mixed, _init_random]
    
    for init_fn in init_methods:
        if time.time() - start_time > DEADLINE:
            break
        
        pos = init_fn()
        score = _score_fast(pos)
        
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
    
    if best_pos is None:
        best_pos = _init_grid()
        best_score = _score_fast(best_pos)
    
    # Simulated annealing
    current_pos = best_pos.copy()
    current_score = best_score
    
    # Temperature schedule
    T_start = 0.05
    T_end = 0.001
    total_iterations = 50000
    
    # Precompute which pairs are close for efficiency
    def _anneal_step(pos, T):
        """Perform one annealing step. Returns new position and score."""
        # Pick a random sphere to move
        i = np.random.randint(N)
        
        # Random displacement
        disp = np.random.normal(0, 0.02, size=3)
        new_pos = pos.copy()
        new_pos[i] = np.clip(pos[i] + disp, -0.5, 0.5)
        
        new_score = _score_fast(new_pos)
        
        if new_score >= current_score:
            return new_pos, new_score, True
        else:
            delta = new_score - current_score
            if np.random.random() < np.exp(delta / T):
                return new_pos, new_score, True
            else:
                return pos, current_score, False
    
    # Run simulated annealing
    it = 0
    while it < total_iterations:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Temperature decay
        T = T_start * (T_end / T_start) ** (it / total_iterations)
        
        new_pos, new_score, accepted = _anneal_step(current_pos, T)
        
        if accepted:
            current_pos = new_pos
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_pos = current_pos.copy()
        
        it += 1
    
    # Final polish: local optimization with smaller steps
    if time.time() - start_time < DEADLINE - 10:
        # Multiple rounds of local improvement
        for round_num in range(3):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            improved = True
            while improved and time.time() - start_time < DEADLINE - 5:
                improved = False
                for i in range(N):
                    if time.time() - start_time > DEADLINE - 5:
                        break
                    
                    # Try moving sphere i in each direction
                    for axis in range(3):
                        for direction in [-1, 1]:
                            for step_size in [0.005, 0.002, 0.001]:
                                trial = best_pos.copy()
                                trial[i, axis] += direction * step_size
                                trial[i, axis] = np.clip(trial[i, axis], -0.5, 0.5)
                                
                                trial_score = _score_fast(trial)
                                if trial_score > best_score + 1e-12:
                                    best_pos = trial
                                    best_score = trial_score
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
    
    return best_pos
