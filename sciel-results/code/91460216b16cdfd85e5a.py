"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def _precompute_aps(n):
    """Precompute all arithmetic progressions in [0, n)."""
    aps = []
    index_to_aps = [[] for _ in range(n)]
    for d in range(1, n + 1):
        for start in range(d):
            idxs = list(range(start, n, d))
            if len(idxs) >= 1:
                ap_id = len(aps)
                aps.append(idxs)
                for idx in idxs:
                    index_to_aps[idx].append(ap_id)
    return aps, index_to_aps

def _compute_max_discrepancy(signs, aps):
    """Compute the maximum absolute signed sum over all APs."""
    max_disc = 0
    for ap in aps:
        s = 0
        for idx in ap:
            s += signs[idx]
        abs_s = abs(s)
        if abs_s > max_disc:
            max_disc = abs_s
    return max_disc

def _incremental_flip(signs, pos, aps, index_to_aps, current_sums):
    """Flip sign at pos and update AP sums. Returns new max discrepancy."""
    old_sign = signs[pos]
    new_sign = -old_sign
    signs[pos] = new_sign
    
    max_disc = 0
    for ap_id in index_to_aps[pos]:
        current_sums[ap_id] += 2 * new_sign  # flip changes sum by 2*new_sign
        abs_s = abs(current_sums[ap_id])
        if abs_s > max_disc:
            max_disc = abs_s
    
    return max_disc

def _compute_all_sums(signs, aps):
    """Compute sum for each AP."""
    sums = []
    for ap in aps:
        s = 0
        for idx in ap:
            s += signs[idx]
        sums.append(s)
    return sums

def construct():
    n = 256
    start_time = time.time()
    time_limit = 58.0  # Safety margin (90s limit, leave 12s)
    
    # Precompute APs
    aps, index_to_aps = _precompute_aps(n)
    num_aps = len(aps)
    
    # Initialize with random signs
    rng = random.Random(12345)
    signs = [rng.choice([-1, 1]) for _ in range(n)]
    
    # Compute initial sums and max discrepancy
    current_sums = _compute_all_sums(signs, aps)
    max_disc = _compute_max_discrepancy(signs, aps)
    best_signs = signs[:]
    best_max_disc = max_disc
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.9997
    steps = 0
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random position to flip
        pos = rng.randint(0, n - 1)
        
        # Try flip
        old_sign = signs[pos]
        new_sign = -old_sign
        signs[pos] = new_sign
        
        # Update affected AP sums
        new_max_disc = 0
        for ap_id in index_to_aps[pos]:
            current_sums[ap_id] += 2 * new_sign
            abs_s = abs(current_sums[ap_id])
            if abs_s > new_max_disc:
                new_max_disc = abs_s
        
        # Accept or reject
        if new_max_disc <= max_disc or rng.random() < math.exp(-(new_max_disc - max_disc) / T):
            # Accept
            max_disc = new_max_disc
            if max_disc < best_max_disc:
                best_max_disc = max_disc
                best_signs = signs[:]
        else:
            # Reject: revert
            signs[pos] = old_sign
            for ap_id in index_to_aps[pos]:
                current_sums[ap_id] -= 2 * new_sign
        
        # Cool down
        T *= cooling
        steps += 1
        
        # Periodic full recompute to prevent drift
        if steps % 5000 == 0:
            current_sums = _compute_all_sums(signs, aps)
            max_disc = _compute_max_discrepancy(signs, aps)
    
    # Final local search polish
    if time.time() - start_time < time_limit - 5:
        # Greedy improvement pass
        improved = True
        while improved and time.time() - start_time < time_limit - 5:
            improved = False
            for pos in range(n):
                old_sign = signs[pos]
                new_sign = -old_sign
                signs[pos] = new_sign
                
                new_max_disc = 0
                for ap_id in index_to_aps[pos]:
                    current_sums[ap_id] += 2 * new_sign
                    abs_s = abs(current_sums[ap_id])
                    if abs_s > new_max_disc:
                        new_max_disc = abs_s
                
                if new_max_disc < max_disc:
                    max_disc = new_max_disc
                    improved = True
                    if max_disc < best_max_disc:
                        best_max_disc = max_disc
                        best_signs = signs[:]
                else:
                    # Revert
                    signs[pos] = old_sign
                    for ap_id in index_to_aps[pos]:
                        current_sums[ap_id] -= 2 * new_sign
    
    # Verify and return best
    final_sums = _compute_all_sums(best_signs, aps)
    final_max = max(abs(s) for s in final_sums)
    
    return best_signs
# EVOLVE-BLOCK-END
