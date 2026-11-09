"""Improved program for low discrepancy on arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # 5s margin before 60s limit
    
    def compute_max_abs_sum_fast(signs):
        """
        Computes the maximum absolute signed sum over all arithmetic progressions.
        Optimized for speed by early exiting if max_abs >= 3 (since we want low values).
        """
        max_abs = 0
        s = signs
        nn = n
        # We check all steps k from 1 to nn-1
        # For each k, we check starts i from 0 to k-1
        # For each start, we accumulate the sum
        for k in range(1, nn):
            for i in range(k):
                curr = 0
                idx = i
                while idx < nn:
                    curr += s[idx]
                    abs_curr = curr if curr > 0 else -curr
                    if abs_curr > max_abs:
                        max_abs = abs_curr
                        if max_abs >= 4:
                            return max_abs
                    idx += k
        return max_abs

    def local_search(signs, current_max, max_passes=5, time_limit=None):
        """
        Performs local search by flipping signs that reduce the max absolute sum.
        Returns (new_signs, new_max).
        """
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time_limit and (time.time() > time_limit):
                break
            improved = False
            passes += 1
            for pos in range(n):
                if time_limit and (time.time() > time_limit):
                    break
                old_val = signs[pos]
                signs[pos] = -old_val
                new_max = compute_max_abs_sum_fast(signs)
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                else:
                    signs[pos] = old_val
        return signs, current_max

    rng = random.Random(42)
    
    # 1. Initial Random Start and Local Search
    best_signs = [rng.choice((-1, 1)) for _ in range(n)]
    best_max = compute_max_abs_sum_fast(best_signs)
    
    # Run a few passes of local search on the initial candidate
    # Limit time for this phase to 5 seconds
    phase1_limit = start_time + 5
    best_signs, best_max = local_search(best_signs, best_max, max_passes=10, time_limit=phase1_limit)
    
    # 2. Random Restarts
    # Use remaining time for random restarts
    # Each restart gets a small budget
    restarts_done = 0
    max_restarts = 20  # Hard cap to prevent infinite loops if time check fails
    
    while (time.time() < deadline - 2) and restarts_done < max_restarts:
        restarts_done += 1
        candidate = [rng.choice((-1, 1)) for _ in range(n)]
        cand_max = compute_max_abs_sum_fast(candidate)
        
        # If already better, great. If not, try to improve it slightly.
        # Limit local search for each restart to 2 passes or 1 second
        restart_limit = min(time.time() + 1.0, deadline - 2)
        candidate, cand_max = local_search(candidate, cand_max, max_passes=2, time_limit=restart_limit)
        
        if cand_max < best_max:
            best_max = cand_max
            best_signs = candidate[:]
            
    return best_signs
# EVOLVE-BLOCK-END
