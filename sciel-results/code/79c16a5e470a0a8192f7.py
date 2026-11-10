"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256
_DEADLINE = 85.0  # safety margin before 90s kill

def _max_abs_ap_sum(signs):
    """Compute max |sum| over all arithmetic progressions in [1, N]."""
    n = len(signs)
    signs = np.asarray(signs, dtype=np.int32)
    max_disc = 1  # singletons always give |sum| = 1
    
    # For each step d and residue r mod d, APs are contiguous subsequences of signs[r::d]
    # We need max over all contiguous subsequence sums
    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            if len(seq) <= 1:
                continue
            csum = np.cumsum(seq)
            # max subarray sum and min subarray sum
            # max subarray: max(csum) - min(csum[:i]) for each i
            # Use Kadane-like via prefix
            min_prefix = np.minimum.accumulate(np.concatenate(([0], csum)))
            max_prefix = np.maximum.accumulate(np.concatenate(([0], csum)))
            # max subarray sum = max(csum - min_prefix[:-1])
            max_sub = np.max(csum - min_prefix[:-1])
            min_sub = np.min(csum - max_prefix[:-1])
            candidate = max(abs(max_sub), abs(min_sub))
            if candidate > max_disc:
                max_disc = candidate
    return max_disc

def _greedy_construct(rng):
    """Greedy construction: add one sign at a time to minimize current max violation."""
    signs = []
    # Precompute which APs each position participates in
    # Position i (0-indexed) is in APs (start, step, len) where start <= i < start + (len-1)*step
    # For greedy, we just try both -1 and +1 and pick the one with lower max violation
    
    for i in range(_N):
        best_sign = None
        best_val = float('inf')
        for s in (1, -1):
            trial = signs + [s]
            # Only need to check APs that include position i
            # For efficiency, compute max over all APs but only those ending at or including i
            # Actually for simplicity in early stages, just compute full max
            # But that's O(N^3) total... let's be smarter
            # For position i, relevant APs are those where i is in [start, start+(len-1)*step]
            # i.e., start <= i and (i - start) % step == 0 and i - start < (len-1)*step
            # For greedy with small i, just compute max over APs involving position i
            signs_arr = np.array(trial, dtype=np.int32)
            n_trial = len(trial)
            max_v = 1
            for d in range(1, n_trial):
                for r in range(d):
                    seq = signs_arr[r::d]
                    if len(seq) <= 1:
                        continue
                    # Check if position i is in this sequence
                    # Position i is in seq if i % d == r
                    if i % d == r:
                        csum = np.cumsum(seq)
                        min_prefix = np.minimum.accumulate(np.concatenate(([0], csum)))
                        max_prefix = np.maximum.accumulate(np.concatenate(([0], csum)))
                        max_sub = np.max(csum - min_prefix[:-1])
                        min_sub = np.min(csum - max_prefix[:-1])
                        candidate = max(abs(max_sub), abs(min_sub))
                        if candidate > max_v:
                            max_v = candidate
            if max_v < best_val:
                best_val = max_v
                best_sign = s
        signs.append(best_sign)
    return signs

def _local_search(signs, rng, deadline):
    """Local search: flip signs to reduce max |AP sum|."""
    signs = list(signs)
    current_max = _max_abs_ap_sum(signs)
    best_signs = list(signs)
    best_max = current_max
    start_time = time.time()
    
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() - start_time < deadline:
        if no_improve_count > max_no_improve:
            # Random restart within local search
            for _ in range(10):
                idx = rng.randint(0, _N - 1)
                signs[idx] = -signs[idx]
            current_max = _max_abs_ap_sum(signs)
            no_improve_count = 0
            continue
        
        # Pick a random position to flip
        idx = rng.randint(0, _N - 1)
        old_sign = signs[idx]
        signs[idx] = -old_sign
        new_max = _max_abs_ap_sum(signs)
        
        if new_max < current_max:
            current_max = new_max
            no_improve_count = 0
            if current_max < best_max:
                best_max = current_max
                best_signs = list(signs)
        else:
            signs[idx] = old_sign
            no_improve_count += 1
    
    return best_signs, best_max

def construct():
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Phase 1: Greedy construction
    signs = _greedy_construct(rng)
    current_max = _max_abs_ap_sum(signs)
    best_signs = list(signs)
    best_max = current_max
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_signs
    
    # Phase 2: Local search
    best_signs, best_max = _local_search(best_signs, rng, deadline - start_time)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_signs
    
    # Phase 3: Try a few more random restarts with local search if time permits
    for restart in range(5):
        if time.time() - start_time > deadline:
            break
        # Perturb best solution
        trial = list(best_signs)
        for _ in range(rng.randint(5, 20)):
            idx = rng.randint(0, _N - 1)
            trial[idx] = -trial[idx]
        
        trial_signs, trial_max = _local_search(trial, rng, deadline - start_time)
        if trial_max < best_max:
            best_max = trial_max
            best_signs = trial_signs
    
    return best_signs
# EVOLVE-BLOCK-END
