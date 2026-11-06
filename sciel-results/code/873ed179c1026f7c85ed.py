"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256
DEADLINE = 85.0  # safety margin before 90s kill

def _precompute_aps():
    """
    Precompute all arithmetic progressions in [1, N] (1-indexed) with length >= 2.
    Returns a list of lists, each inner list is 0-indexed positions.
    We also need to consider singletons (length 1), but their sum is always ±1,
    so the max absolute sum is at least 1. We focus on length >= 2 for optimization,
    but the final metric includes singletons. Since singletons contribute max 1,
    and we expect the max over longer APs to be > 1, we can ignore singletons
    in the search and just ensure the returned answer is valid.
    Actually, the task says "including one". The max absolute sum will be max(1, max_AP_sum).
    Since we are minimizing the max, and 1 is a lower bound, we just need to minimize
    the max over APs of length >= 2, and the final answer will be max(1, that value).
    But wait, if all APs of length >= 2 have sum 0, the max is 1.
    So we minimize the max over APs of length >= 2.
    """
    aps = []
    for step in range(1, N // 2 + 1):
        for start in range(1, N + 1):
            # The AP is start, start+step, start+2*step, ...
            # We need at least 2 terms: start + step <= N
            if start + step > N:
                break
            # Generate the AP
            positions = []
            val = start
            while val <= N:
                positions.append(val - 1)  # 0-indexed
                val += step
            if len(positions) >= 2:
                aps.append(positions)
    return aps

def _compute_max_abs_sum(signs, aps):
    """
    Compute the maximum absolute signed sum over all APs.
    signs: array of length N with values -1 or 1.
    aps: list of lists of 0-indexed positions.
    """
    max_abs = 0
    for ap in aps:
        s = np.sum(signs[ap])
        abs_s = abs(s)
        if abs_s > max_abs:
            max_abs = abs_s
    return max_abs

def _incremental_max_abs(signs, aps, current_max, changed_idx):
    """
    Update the max absolute sum after flipping a sign at changed_idx.
    This is a heuristic; for accuracy, we might just recompute if the number of
    APs containing changed_idx is small, or use a smarter update.
    However, recomputing is O(num_aps * avg_len). With ~10k APs and avg_len ~5,
    it's ~50k ops. This is fast.
    """
    return _compute_max_abs_sum(signs, aps)

def construct():
    start_time = time.time()
    
    # Precompute APs
    aps = _precompute_aps()
    num_aps = len(aps)
    
    # Initialize signs
    rng = random.Random(171)
    signs = np.array([rng.choice((-1, 1)) for _ in range(N)], dtype=np.int8)
    
    # Initial evaluation
    current_max = _compute_max_abs_sum(signs, aps)
    best_signs = signs.copy()
    best_max = current_max
    
    # Local search: flip signs to reduce the max
    # We prioritize flipping signs that are part of the "worst" APs.
    
    # To speed up, we can precompute for each index, the list of APs it belongs to.
    # But that's memory intensive. Instead, we can just evaluate the effect of a flip
    # by checking all APs.
    
    # Let's use a simpler local search: try flipping each index, if it reduces the max, keep it.
    # Repeat until no improvement or deadline.
    
    improved = True
    while improved and (time.time() - start_time) < DEADLINE:
        improved = False
        for i in range(N):
            if (time.time() - start_time) > DEADLINE - 1:
                break
            # Try flipping index i
            old_sign = signs[i]
            signs[i] = -old_sign
            new_max = _compute_max_abs_sum(signs, aps)
            if new_max < best_max:
                best_max = new_max
                best_signs = signs.copy()
                improved = True
            else:
                signs[i] = old_sign
    
    # If we still have time, do a more aggressive search:
    # Random restarts or simulated annealing.
    # Given the time budget, let's try a few random restarts with local search.
    while (time.time() - start_time) < DEADLINE - 5:
        # Random restart
        signs = np.array([rng.choice((-1, 1)) for _ in range(N)], dtype=np.int8)
        current_max = _compute_max_abs_sum(signs, aps)
        
        # Local search
        for _ in range(10):  # Limited iterations for speed
            if (time.time() - start_time) > DEADLINE - 5:
                break
            improved = False
            for i in range(N):
                old_sign = signs[i]
                signs[i] = -old_sign
                new_max = _compute_max_abs_sum(signs, aps)
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                else:
                    signs[i] = old_sign
                if not improved:
                    break
            if not improved:
                break
        
        if current_max < best_max:
            best_max = current_max
            best_signs = signs.copy()
    
    # Convert to list of Python integers
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
