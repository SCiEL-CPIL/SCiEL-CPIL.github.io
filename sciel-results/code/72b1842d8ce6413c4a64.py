"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

_N = 256
_DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _max_abs_ap_sum(signs):
    """Compute the maximum absolute signed sum over all arithmetic progressions in [1,n]."""
    n = len(signs)
    signs = np.asarray(signs, dtype=np.int32)
    max_disc = 1  # singletons always give |sum| = 1
    
    # For each step d and residue r mod d, compute prefix sums of signs[r::d]
    # The APs with step d and starting at positions congruent to r mod d
    # are contiguous subsequences of signs[r::d].
    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            if len(seq) > 1:
                csum = np.cumsum(seq)
                # All contiguous subsequence sums are csum[j] - csum[i-1] for i<=j
                # Max subarray sum and min subarray sum give max abs
                max_sub = np.max(csum) - np.min(csum)
                # Actually we need max |sum| which is max(max_sub, -min_sub)
                # But max_sub already captures the range. Let's think more carefully.
                # The sum from index i to j is csum[j] - csum[i-1] (with csum[-1]=0)
                # So we need max over all i<=j of |csum[j] - csum[i-1]|
                # This equals max(csum) - min(csum) if we include the implicit 0 at the start
                # Actually, let's use a different approach:
                # Prepend 0 to csum, then max diff between any two elements where later >= earlier in position
                # But since we want absolute value, it's just max(csum_ext) - min(csum_ext)
                # where csum_ext = [0] + list(csum)
                csum_ext = np.concatenate(([0], csum))
                disc = int(np.max(csum_ext) - np.min(csum_ext))
                if disc > max_disc:
                    max_disc = disc
    return max_disc

def _thue_morse_init():
    """Generate Thue-Morse sequence of length _N mapped to +/-1."""
    signs = []
    for i in range(_N):
        # Count number of 1-bits in i, map even -> +1, odd -> -1
        bits = bin(i).count('1')
        signs.append(1 if bits % 2 == 0 else -1)
    return signs

def _local_search(signs, time_limit):
    """Perform local search to minimize max abs AP sum."""
    n = len(signs)
    signs = list(signs)
    current_disc = _max_abs_ap_sum(signs)
    best_signs = signs[:]
    best_disc = current_disc
    
    start_time = time.time()
    
    # Precompute AP membership for each position to speed up flip evaluation
    # For each position p, store which APs contain p
    # This is complex; instead, we'll do full recomputation but limit steps
    
    rng = random.Random(42)
    steps = 0
    
    while time.time() - start_time < time_limit:
        # Try flipping a random position
        p = rng.randint(0, n - 1)
        signs[p] = -signs[p]
        new_disc = _max_abs_ap_sum(signs)
        
        if new_disc <= best_disc:
            best_disc = new_disc
            best_signs = signs[:]
        else:
            # Revert if worse (greedy)
            signs[p] = -signs[p]
        
        steps += 1
        if steps % 10000 == 0 and time.time() - start_time > time_limit:
            break
    
    return best_signs, best_disc

def construct():
    start_time = time.time()
    
    # Start with Thue-Morse initialization
    signs = _thue_morse_init()
    current_disc = _max_abs_ap_sum(signs)
    best_signs = signs[:]
    best_disc = current_disc
    
    # Time budget for local search
    remaining = _DEADLINE - (time.time() - start_time)
    if remaining > 1:
        best_signs, best_disc = _local_search(best_signs, remaining - 2)
    
    # Check deadline
    if time.time() - start_time > _DEADLINE - 2:
        return best_signs
    
    # Try a few random restarts if time permits
    rng = random.Random(123)
    while time.time() - start_time < _DEADLINE - 5:
        # Random initialization
        rand_signs = [rng.choice((-1, 1)) for _ in range(_N)]
        rand_disc = _max_abs_ap_sum(rand_signs)
        
        if rand_disc < best_disc:
            best_signs = rand_signs[:]
            best_disc = rand_disc
        
        # Local search from this point
        remaining = _DEADLINE - 5 - (time.time() - start_time)
        if remaining > 1:
            local_best, local_disc = _local_search(rand_signs, remaining)
            if local_disc < best_disc:
                best_signs = local_best
                best_disc = local_disc
    
    return best_signs
# EVOLVE-BLOCK-END
