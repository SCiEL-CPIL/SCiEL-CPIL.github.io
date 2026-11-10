"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 256
_DEADLINE = 85.0  # seconds, safety margin before 90s kill


def _thue_morse_signs(n):
    """Generate Thue-Morse sequence: -1 for odd popcount, +1 for even popcount."""
    signs = np.ones(n, dtype=np.int32)
    for i in range(n):
        if bin(i).count('1') % 2 == 1:
            signs[i] = -1
    return signs


def _max_abs_ap_sum(signs):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions
    in [1, n] (1-indexed), i.e., over signs[0..n-1] (0-indexed).
    
    An AP is defined by start s (1-indexed), step d >= 1, and length L >= 1.
    The elements are s, s+d, s+2d, ..., s+(L-1)d, all <= n.
    In 0-indexed: positions s-1, s-1+d, ..., s-1+(L-1)d.
    
    For a fixed step d and residue r = (s-1) % d, the APs are contiguous
    subsequences of the subsequence signs[r::d].
    """
    n = len(signs)
    max_disc = 1  # singletons always give |sum| = 1
    
    # For step d=1, all intervals
    csum = np.concatenate(([0], np.cumsum(signs)))
    for i in range(n):
        for j in range(i + 1, n + 1):
            s = csum[j] - csum[i]
            if abs(s) > max_disc:
                max_disc = abs(s)
    
    # For step d >= 2
    for d in range(2, n):
        for r in range(d):
            seq = signs[r::d]
            L = len(seq)
            if L <= 1:
                continue
            csum = np.concatenate(([0], np.cumsum(seq)))
            for i in range(L):
                for j in range(i + 1, L + 1):
                    s = csum[j] - csum[i]
                    if abs(s) > max_disc:
                        max_disc = abs(s)
    
    return max_disc


def _max_abs_ap_sum_fast(signs):
    """
    Faster computation using the fact that for each (d, r), we need the max
    subarray sum (and min subarray sum) of the subsequence signs[r::d].
    The max absolute sum over all contiguous subsequences is max(max_subarray, -min_subarray).
    """
    n = len(signs)
    max_disc = 1  # singletons
    
    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            L = len(seq)
            if L <= 1:
                continue
            # Kadane's algorithm for max and min subarray sum
            max_ending = max_so_far = seq[0]
            min_ending = min_so_far = seq[0]
            for i in range(1, L):
                max_ending = max(seq[i], max_ending + seq[i])
                max_so_far = max(max_so_far, max_ending)
                min_ending = min(seq[i], min_ending + seq[i])
                min_so_far = min(min_so_far, min_ending)
            candidate = max(max_so_far, -min_so_far)
            if candidate > max_disc:
                max_disc = candidate
    
    return max_disc


def _evaluate_flip(signs, pos, ap_index_table):
    """
    Evaluate the change in max_abs_ap_sum if we flip signs[pos].
    ap_index_table: for each position p, list of (d, r, i) where i is the index
    in the subsequence signs[r::d] corresponding to position p.
    
    Actually, let's use a different approach: precompute for each position p,
    which (d, r) pairs include it, and what its index is in that subsequence.
    Then flipping signs[p] changes the subarray sums for all contiguous
    subsequences of signs[r::d] that include index i.
    
    This is complex. Let's just recompute _max_abs_ap_sum_fast after each flip.
    """
    pass


def construct():
    start_time = time.time()
    
    # Initialize with Thue-Morse
    signs = _thue_morse_signs(_N)
    best_signs = signs.copy()
    best_score = _max_abs_ap_sum_fast(signs)
    
    # Local search with random restarts
    import random
    rng = random.Random(42)
    
    while time.time() - start_time < _DEADLINE - 2.0:
        # Try flipping each position
        improved = False
        for pos in range(_N):
            if time.time() - start_time > _DEADLINE - 2.0:
                break
            signs[pos] = -signs[pos]
            new_score = _max_abs_ap_sum_fast(signs)
            if new_score < best_score:
                best_score = new_score
                best_signs = signs.copy()
                improved = True
            else:
                signs[pos] = -signs[pos]  # revert
        
        if not improved:
            # Random restart with perturbation
            if time.time() - start_time > _DEADLINE - 5.0:
                break
            # Perturb current best
            signs = best_signs.copy()
            num_flips = rng.randint(3, 8)
            for _ in range(num_flips):
                p = rng.randint(0, _N - 1)
                signs[p] = -signs[p]
            # Try to improve from here
            for pos in range(_N):
                if time.time() - start_time > _DEADLINE - 2.0:
                    break
                signs[pos] = -signs[pos]
                new_score = _max_abs_ap_sum_fast(signs)
                if new_score < best_score:
                    best_score = new_score
                    best_signs = signs.copy()
                else:
                    signs[pos] = -signs[pos]
    
    return list(best_signs)
# EVOLVE-BLOCK-END
