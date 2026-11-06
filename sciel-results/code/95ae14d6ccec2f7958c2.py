<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    rng=random.Random(171)
    return [rng.choice((-1,1)) for _ in range(256)]
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def compute_max_abs_sum(s):
    """Compute the maximum absolute sum over all arithmetic progressions in [1, 256]."""
    n = len(s)
    max_abs = 0
    for step in range(1, n + 1):
        for start in range(1, n + 1):
            if start > n:
                break
            total = 0
            idx = start - 1
            while idx < n:
                total += s[idx]
                max_abs = max(max_abs, abs(total))
                idx += step
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Start with a deterministic sequence that might be good
    # Try a few different initial patterns
    best_seq = None
    best_score = float('inf')
    
    # Try a simple alternating pattern first
    seq = [1 if i % 2 == 0 else -1 for i in range(256)]
    score = compute_max_abs_sum(seq)
    if score < best_score:
        best_score = score
        best_seq = seq[:]
    
    # Try a few other deterministic patterns
    patterns = [
        [1] * 256,
        [-1] * 256,
        [1 if i % 4 < 2 else -1 for i in range(256)],
        [1 if i % 3 < 2 else -1 for i in range(256)],
    ]
    
    for p in patterns:
        if time.time() > deadline:
            break
        score = compute_max_abs_sum(p)
        if score < best_score:
            best_score = score
            best_seq = p[:]
    
    # Now do local search: flip individual bits and see if it improves
    current = best_seq[:]
    current_score = best_score
    
    # Precompute for faster evaluation: we'll use a smarter approach
    # For each position, we can quickly estimate the impact of flipping
    
    rng = np.random.RandomState(42)
    
    # Local search with random restarts
    while time.time() < deadline:
        # Try flipping each position
        improved = False
        for i in range(256):
            if time.time() > deadline:
                break
            current[i] = -current[i]
            new_score = compute_max_abs_sum(current)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_seq = current[:]
                improved = True
            else:
                current[i] = -current[i]  # revert
        
        # If no improvement, try a random perturbation
        if not improved:
            # Flip a few random positions
            num_flips = rng.randint(1, 5)
            for _ in range(num_flips):
                i = rng.randint(0, 255)
                current[i] = -current[i]
            new_score = compute_max_abs_sum(current)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_seq = current[:]
            else:
                # Revert
                for _ in range(num_flips):
                    i = rng.randint(0, 255)
                    current[i] = -current[i]
    
    return best_seq
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
