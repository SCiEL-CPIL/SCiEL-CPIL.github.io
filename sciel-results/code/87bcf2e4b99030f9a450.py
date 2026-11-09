"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 55.0  # safety margin before 60s kill
    
    # Precompute all arithmetic progressions
    # For each step d >= 1 and start s, collect all valid APs
    # An AP is defined by (start, step, length) where start + (length-1)*step < n
    # We need to evaluate max |sum| over any contiguous subsequence of each AP
    
    # Build list of APs as (start_idx, step, length)
    # For efficiency, we'll store them as arrays
    ap_starts = []
    ap_steps = []
    ap_lengths = []
    
    for d in range(1, n + 1):
        for s in range(n):
            # Find max length: s + (L-1)*d < n => L-1 < (n-s)/d => L <= floor((n-1-s)/d) + 1
            max_L = (n - 1 - s) // d + 1
            if max_L >= 1:
                # We only need to consider the full AP up to max_L
                # But for evaluation, we need all subsequences
                # Actually, for a fixed (s,d), the AP is s, s+d, s+2d, ...
                # The max abs sum over any contiguous subsequence can be computed
                # using prefix sums of the AP elements
                ap_starts.append(s)
                ap_steps.append(d)
                ap_lengths.append(max_L)
    
    ap_starts = np.array(ap_starts, dtype=np.int32)
    ap_steps = np.array(ap_steps, dtype=np.int32)
    ap_lengths = np.array(ap_lengths, dtype=np.int32)
    
    # Number of APs
    num_aps = len(ap_starts)
    
    # Initialize with Thue-Morse-like sequence
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    def thue_morse(n):
        # Returns 0 or 1
        c = 0
        while n:
            c ^= (n & 1)
            n >>= 1
        return c
    
    x = np.array([1 - 2*thue_morse(i) for i in range(n)], dtype=np.int8)
    
    def evaluate(x_arr):
        """Compute max absolute sum over any AP subsequence."""
        # For each AP, compute prefix sums and find max abs difference
        max_abs = 0
        for i in range(num_aps):
            s = ap_starts[i]
            d = ap_steps[i]
            L = ap_lengths[i]
            # Extract AP elements
            idx = s + d * np.arange(L, dtype=np.int32)
            ap_vals = x_arr[idx]
            # Prefix sums
            prefix = np.cumsum(ap_vals)
            # Max abs sum over any contiguous subsequence = max(prefix) - min(prefix)
            # But we want max |sum|, so it's max(max(prefix), -min(prefix))
            # Actually, max contiguous sum = max(prefix) - min(prefix before)
            # Let's compute properly:
            # min_prefix = minimum prefix sum (including 0)
            # max_prefix = maximum prefix sum
            # max_abs_subseq = max(max_prefix, -min_prefix)
            min_pref = np.minimum.accumulate(np.concatenate(([0], prefix)))
            max_pref = np.maximum.accumulate(np.concatenate(([0], prefix)))
            # The max abs contiguous sum is max(max_pref, -min_pref)
            cur_max = max(max_pref[-1], -min_pref[-1])
            if cur_max > max_abs:
                max_abs = cur_max
        return max_abs
    
    # Simulated annealing
    rng = random.Random(123)
    best_x = x.copy()
    best_score = evaluate(x)
    current_score = best_score
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline - 2.0:
        # Pick a random position to flip
        pos = rng.randint(0, n - 1)
        x[pos] = -x[pos]
        
        new_score = evaluate(x)
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            x[pos] = -x[pos]
        
        T *= cooling
    
    return best_x.tolist()
# EVOLVE-BLOCK-END
