"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256
_DEADLINE = 82.0  # safety margin before 90s kill

def _max_abs_ap_sum(signs):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions
    in [1, n] with positive step and length >= 1.
    signs: numpy array of -1/+1 of length n.
    Returns (max_abs, worst_ap) where worst_ap is (start, step, length) 1-indexed.
    """
    n = len(signs)
    max_abs = 1  # singletons
    worst_ap = (1, 1, 1)
    
    for d in range(1, n):
        for r in range(d):
            # positions r, r+d, r+2d, ... (0-indexed)
            seq = signs[r::d]
            if len(seq) <= 1:
                continue
            csum = np.cumsum(seq)
            # all contiguous subarray sums
            # sum(i..j) = csum[j] - csum[i-1] (with csum[-1]=0)
            # We want max |sum(i..j)| for 0<=i<=j<len
            # = max(csum) - min(csum_prefix) where we consider prefix sums including 0
            prefix = np.empty(len(csum) + 1)
            prefix[0] = 0
            prefix[1:] = csum
            max_sub = np.max(prefix) - np.min(prefix)
            if max_sub > max_abs:
                max_abs = int(max_sub)
                # Find the worst AP for potential repair
                # Find i,j such that |csum[j] - csum[i-1]| is max
                # prefix[i] = csum[i-1] for i>=1, prefix[0]=0
                min_idx = np.argmin(prefix)
                max_idx = np.argmax(prefix)
                if max_idx > min_idx:
                    i, j = min_idx, max_idx
                else:
                    i, j = max_idx, min_idx
                # i is the "start index in prefix" which corresponds to position i in seq (0-indexed)
                # j is the "end index in prefix" which corresponds to position j-1 in seq
                # AP in seq: from index i to index j-1 (inclusive), length = j - i
                # In original: start = r + i*d, length = j - i
                start_1idx = r + i * d + 1
                length = j - i
                worst_ap = (start_1idx, d, length)
    return max_abs, worst_ap

def _greedy_construct():
    """
    Greedy construction: build signs one by one, choosing the sign that
    minimizes the current maximum absolute AP sum.
    """
    n = _N
    signs = np.zeros(n, dtype=np.int32)
    
    # We'll build incrementally. At each step, try both signs and pick the better one.
    # To speed up, we only check APs that end at the current position or are affected.
    # Actually, for simplicity and correctness, let's do a full check but it's slow.
    # Better: use a heuristic. Let's try a different approach.
    
    # Alternative: start with random, then local search.
    # Or: use a known good pattern and perturb.
    
    # Let's use a simple heuristic: Thue-Morse or similar.
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n.
    # Map to -1/+1.
    
    tm = np.array([(-1) ** bin(i).count('1') for i in range(n)], dtype=np.int32)
    val, _ = _max_abs_ap_sum(tm)
    
    # Also try a random start
    rng = random.Random(42)
    best = tm.copy()
    best_val = val
    
    for _ in range(20):
        cand = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int32)
        v, _ = _max_abs_ap_sum(cand)
        if v < best_val:
            best_val = v
            best = cand.copy()
    
    return best, best_val

def _local_search(signs, time_limit=75.0):
    """
    Local search: flip single positions to reduce max_abs_ap_sum.
    Uses efficient delta computation.
    """
    n = len(signs)
    signs = signs.copy()
    current_val, worst_ap = _max_abs_ap_sum(signs)
    start_time = time.time()
    no_improve = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try flipping positions in the worst AP first
        s, d, length = worst_ap
        # Positions in the AP (0-indexed): s-1, s-1+d, ..., s-1+(length-1)*d
        ap_positions = list(range(s - 1, s - 1 + length * d, d))
        
        # Also try some random positions
        rng = random.Random(int(time.time() * 1000) % 10000)
        trial_positions = ap_positions + [rng.randrange(n) for _ in range(50)]
        # Remove duplicates
        trial_positions = list(set(trial_positions))
        rng.shuffle(trial_positions)
        
        for pos in trial_positions:
            if time.time() - start_time >= time_limit:
                break
            
            old_sign = signs[pos]
            signs[pos] = -old_sign
            
            new_val, new_worst = _max_abs_ap_sum(signs)
            
            if new_val < current_val:
                current_val = new_val
                worst_ap = new_worst
                improved = True
                no_improve = 0
            else:
                signs[pos] = old_sign
                no_improve += 1
        
        if not improved and no_improve > max_no_improve:
            # Random restart within local search
            for _ in range(100):
                pos = random.randrange(n)
                signs[pos] = -signs[pos]
            current_val, worst_ap = _max_abs_ap_sum(signs)
            no_improve = 0
    
    return signs, current_val

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction
    signs, val = _greedy_construct()
    
    # Phase 2: Local search
    if time.time() - start_time < _DEADLINE - 5:
        signs, val = _local_search(signs, time_limit=_DEADLINE - 5 - (time.time() - start_time))
    
    # Ensure we return a list of Python integers
    return [int(s) for s in signs]

# EVOLVE-BLOCK-END
