"""Rewritten program: local search minimizing max |sum| over all APs in [1,256]."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256

def _max_abs_ap_sum_numpy(signs):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions
    contained in indices 0..n-1 using NumPy for speed.
    """
    n = len(signs)
    signs = np.asarray(signs, dtype=np.int32)
    max_disc = 1  # singletons
    
    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            if len(seq) > 1:
                csum = np.cumsum(seq)
                m = np.max(np.abs(csum))
                if m > max_disc:
                    max_disc = m
    return max_disc

def _local_search(signs, max_time, rng):
    """
    Local search: flip one sign at a time if it reduces the max AP sum.
    """
    n = len(signs)
    signs = list(signs)
    current_score = _max_abs_ap_sum_numpy(signs)
    start = time.time()
    
    while time.time() - start < max_time:
        improved = False
        # Try flipping each position
        for i in range(n):
            signs[i] = -signs[i]
            new_score = _max_abs_ap_sum_numpy(signs)
            if new_score < current_score:
                current_score = new_score
                improved = True
                break
            else:
                signs[i] = -signs[i]  # revert
        if not improved:
            # Stuck in local minimum; try a random restart perturbation
            for _ in range(10):
                idx = rng.randint(0, n-1)
                signs[idx] = -signs[idx]
            new_score = _max_abs_ap_sum_numpy(signs)
            if new_score < current_score:
                current_score = new_score
            else:
                # Revert perturbation
                signs[idx] = -signs[idx]
                break
    
    return signs, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second safety margin
    
    rng = random.Random(171)
    
    # Start with random signs
    best_signs = [rng.choice((-1, 1)) for _ in range(_N)]
    best_score = _max_abs_ap_sum_numpy(best_signs)
    
    # Local search
    signs, score = _local_search(best_signs, 75, rng)
    if score < best_score:
        best_score = score
        best_signs = signs
    
    # If we have time, try a few more random restarts
    while time.time() < deadline - 5:
        new_signs = [rng.choice((-1, 1)) for _ in range(_N)]
        new_score = _max_abs_ap_sum_numpy(new_signs)
        if new_score < best_score:
            best_score = new_score
            best_signs = new_signs
        
        # Quick local search on this new start
        signs, score = _local_search(new_signs, min(10, deadline - 5 - time.time()), rng)
        if score < best_score:
            best_score = score
            best_signs = signs
    
    return best_signs
# EVOLVE-BLOCK-END
