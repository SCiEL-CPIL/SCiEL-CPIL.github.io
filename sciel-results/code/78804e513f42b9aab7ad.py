"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _compute_max_discrepancy(signs):
    """Compute maximum absolute signed sum over all arithmetic progressions in [1, n]."""
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    max_disc = 1  # Singletons have abs sum 1
    
    # Iterate over all possible steps
    for step in range(1, n + 1):
        # For a given step, the arithmetic progressions are determined by their start index r.
        # The valid starts are r such that r < n.
        # The sequence for start r is arr[r], arr[r+step], arr[r+2*step], ...
        for r in range(step):
            if r >= n:
                break
            subseq = arr[r::step]
            if len(subseq) == 0:
                continue
            # Compute prefix sums to find max absolute sum over all prefixes
            prefix_sums = np.cumsum(subseq)
            current_max = np.max(np.abs(prefix_sums))
            if current_max > max_disc:
                max_disc = int(current_max)
    
    return max_disc

def _simulated_annealing(signs, time_limit=55.0):
    """Perform Simulated Annealing to minimize maximum absolute signed sum."""
    n = len(signs)
    signs = list(signs)
    current_disc = _compute_max_discrepancy(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    start_time = time.time()
    rng = random.Random(42)
    
    # Initial temperature
    T = 10.0
    # Cooling rate
    alpha = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick a random position to flip
        idx = rng.randint(0, n - 1)
        old_sign = signs[idx]
        signs[idx] = -old_sign
        
        new_disc = _compute_max_discrepancy(signs)
        delta = new_disc - current_disc
        
        if delta < 0 or (T > 0.01 and rng.random() < np.exp(-delta / T)):
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs.copy()
        else:
            # Revert
            signs[idx] = old_sign
            
        T *= alpha
        
    return best_signs, best_disc

def construct():
    n = 256
    start_time = time.time()
    deadline = 58.0  # Safety margin before 60s
    
    # Initial random start
    rng = random.Random(171)
    signs = [rng.choice((-1, 1)) for _ in range(n)]
    
    # Run Simulated Annealing
    best_signs, best_disc = _simulated_annealing(signs, time_limit=deadline - (time.time() - start_time))
    
    # Check time, if still have time, try more restarts
    elapsed = time.time() - start_time
    if elapsed < deadline:
        # Try a few more restarts if time permits
        for restart in range(10):
            if time.time() - start_time > deadline:
                break
            signs = [rng.choice((-1, 1)) for _ in range(n)]
            new_signs, new_disc = _simulated_annealing(signs, time_limit=deadline - (time.time() - start_time))
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = new_signs
    
    return best_signs
# EVOLVE-BLOCK-END
