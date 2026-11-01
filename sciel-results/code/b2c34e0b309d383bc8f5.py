"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _score(c):
    """Compute the largest common radius for centres c (shape (N,3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # distance to walls
    wall = np.min(np.min(0.5 - np.abs(c), axis=1), axis=0)
    # pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    return min(wall, min_dist / 2.0)

def _init_grid():
    """Simple cubic grid as starting point."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _init_random():
    """Random initialisation inside the cube."""
    return np.random.uniform(-0.5, 0.5, size=(N, 3))

# ---------------------------------------------------------------------------
# Simulated Annealing Strategy
# ---------------------------------------------------------------------------

def _anneal(c, time_limit, initial_temp=0.05, final_temp=0.0001, cooling=0.9995):
    """
    Simulated annealing to maximize the minimum distance (radius).
    Moves:
    1. Perturb a single sphere by a small random vector.
    2. Swap two spheres (to break symmetry).
    3. Nudge a sphere towards/away from a wall or neighbor to fix violations.
    """
    c = c.copy()
    best_c = c.copy()
    best_score = _score(best_c)
    
    start_time = time.time()
    temp = initial_temp
    
    # Precompute some constants
    max_coord = 0.5
    
    while time.time() < time_limit:
        # Check time frequently to avoid long loops
        if time.time() > time_limit:
            break
            
        # Choose a move type
        move_type = np.random.rand()
        
        if move_type < 0.7:
            # Move 1: Perturb a single sphere
            i = np.random.randint(N)
            # Step size proportional to temperature
            step = temp * 2.0
            delta = np.random.normal(0, step, size=3)
            
            old_pos = c[i].copy()
            new_pos = old_pos + delta
            new_pos = np.clip(new_pos, -max_coord, max_coord)
            
            if np.allclose(old_pos, new_pos):
                continue
                
            c[i] = new_pos
            
            new_score = _score(c)
            
            # Accept if better or with probability exp((new-old)/T)
            if new_score > best_score:
                best_score = new_score
                best_c = c.copy()
            elif new_score >= best_score - 1e-12:
                # Accept if not significantly worse (plateau)
                pass
            else:
                # Metropolis criterion
                delta_e = new_score - _score(best_c) # Approximate current energy diff
                # Actually we need current score, not best score for acceptance
                curr_score = new_score
                # We didn't store curr_score before move, so let's just use simple hill climbing with random restarts or proper SA state
                # For proper SA, we need current state score.
                # Let's refactor to keep current_score
                pass
            
            # Correct SA logic:
            # We need to track current score.
            # Let's restart this block with proper state tracking.
            
        # The above inline logic is messy. Let's implement a cleaner SA loop.
        # Breaking out to rewrite the inner loop properly.
        # Actually, I should just write the whole function correctly.
        
    return best_c

def _anneal_correct(c, time_limit, initial_temp=0.02, final_temp=0.00005, cooling=0.9998):
    """
    Proper Simulated Annealing implementation.
    """
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    start_time = time.time()
    temp = initial_temp
    
    max_coord = 0.5
    
    while time.time() < time_limit:
        if temp < final_temp:
            temp = final_temp
            
        # Select move
        move_type = np.random.rand()
        
        old_pos = None
        old_pos_j = None
        i = -1
        j = -1
        
        if move_type < 0.8:
            # Perturb single sphere
            i = np.random.randint(N)
            old_pos = c[i].copy()
            step = temp * 1.5
            delta = np.random.normal(0, step, size=3)
            c[i] = np.clip(old_pos + delta, -max_coord, max_coord)
        else:
            # Swap two spheres
            i, j = np.random.choice(N, 2, replace=False)
            old_pos = c[i].copy()
            old_pos_j = c[j].copy()
            c[i] = old_pos_j
            c[j] = old_pos
            
        new_score = _score(c)
        
        # Acceptance
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_c = c.copy()
        else:
            # Metropolis
            delta_e = new_score - current_score
            if np.random.rand() < math.exp(delta_e / temp):
                current_score = new_score
            else:
                # Revert
                if move_type < 0.8:
                    c[i] = old_pos
                else:
                    c[i] = old_pos
                    c[j] = old_pos_j
                    
        # Cool down
        temp *= cooling
        
    return best_c

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Start with a good grid initialization
    best_c = _init_grid()
    best_score = _score(best_c)
    
    # Run simulated annealing from the grid start
    t1 = time.time()
    # Allow most of the time for SA
    sa_time_limit = deadline - 5.0 
    c_sa = _anneal_correct(best_c, sa_time_limit)
    sc_sa = _score(c_sa)
    
    if sc_sa > best_score:
        best_score = sc_sa
        best_c = c_sa
    
    # If we have time, try a random start SA
    if time.time() < deadline - 10.0:
        c_rand = _init_random()
        t2 = time.time()
        remaining = deadline - 10.0 - t2
        if remaining > 5.0:
            c_rand_sa = _anneal_correct(c_rand, remaining)
            sc_rand = _score(c_rand_sa)
            if sc_rand > best_score:
                best_score = sc_rand
                best_c = c_rand_sa
                
    # Final check and return
    return best_c
# EVOLVE-BLOCK-END
