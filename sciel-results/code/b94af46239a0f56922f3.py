"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = 256
    # We use a deterministic construction based on the Thue-Morse sequence,
    # which is known to have low discrepancy properties.
    # The Thue-Morse sequence t(n) is defined as (-1)^(popcount(n)).
    # We assign signs based on t(i) for i in 0..255 (corresponding to positions 1..256).
    
    # Generate Thue-Morse sequence of length 256
    signs = []
    for i in range(n):
        # Count number of 1s in binary representation of i
        c = bin(i).count('1')
        signs.append(1 if c % 2 == 0 else -1)
    
    # However, pure Thue-Morse might not be optimal for all arithmetic progressions in [1, n].
    # Let's try a local search improvement on top of a good initial guess.
    # But we have 60 seconds. Let's see if we can do a quick simulated annealing or hill climbing.
    
    # Actually, let's first check the discrepancy of Thue-Morse.
    # If it's already good, we might just return it.
    # Let's implement a fast check function.
    
    def max_discrepancy(signs_list):
        n = len(signs_list)
        max_abs = 0
        # Iterate over all arithmetic progressions
        # Start s from 1 to n
        # Step d from 1 to n
        # Length k from 1 to floor((n - s) / d) + 1
        # Sum signs[s-1 : s-1+k*d : d]
        
        # This is O(n^3) which is 256^3 ~ 16 million, which is too slow in Python if done naively.
        # We need a faster way or a smarter search.
        
        # Let's optimize the check.
        # For a fixed step d, we can compute prefix sums for each residue class mod d.
        # There are d residue classes. For each residue class r, the elements are r, r+d, r+2d, ...
        # We can compute prefix sums for this sequence.
        # Then for any start s and length k, the sum is prefix[s//d + k] - prefix[s//d].
        # Wait, the indices are 1-based in the problem description [1, 256].
        # Let's use 0-based indexing internally.
        # Position i (0-based) corresponds to value i+1.
        # Arithmetic progression: start s (1-based), step d, length k.
        # Elements: s, s+d, s+2d, ..., s+(k-1)d.
        # 0-based indices: s-1, s-1+d, ..., s-1+(k-1)d.
        
        for d in range(1, n + 1):
            # For each residue class mod d
            for r in range(d):
                # Elements with index i such that i % d == r
                # These are r, r+d, r+2d, ...
                # Let's extract these signs
                seq = []
                for i in range(r, n, d):
                    seq.append(signs_list[i])
                # Compute prefix sums
                prefix = [0]
                for val in seq:
                    prefix.append(prefix[-1] + val)
                # Now, any arithmetic progression with step d and start s (1-based)
                # corresponds to a contiguous subsequence in this `seq` array.
                # The start index in `seq` is (s-1) // d.
                # Let j = (s-1) // d.
                # The length k can be from 1 to len(seq) - j.
                # We want to find the max absolute sum of any contiguous subsequence in `seq`.
                # This is the maximum subarray sum (Kadane's algorithm variant for max absolute).
                # We need max |sum| over all contiguous subarrays.
                
                # Let's compute max positive and max negative sum.
                max_pos = 0
                max_neg = 0
                cur_pos = 0
                cur_neg = 0
                for val in seq:
                    cur_pos = max(0, cur_pos + val)
                    max_pos = max(max_pos, cur_pos)
                    
                    cur_neg = min(0, cur_neg + val)
                    max_neg = min(max_neg, cur_neg)
                
                max_abs = max(max_abs, max_pos, -max_neg)
        
        return max_abs

    # Let's check Thue-Morse
    tm_discrepancy = max_discrepancy(signs)
    
    # If Thue-Morse is good enough, we might just use it.
    # But let's try to improve it with a simple hill climbing.
    # We have 60 seconds. The check is O(n^2 * n/d) ~ O(n^2 log n) or O(n^2) roughly.
    # For n=256, n^2 = 65536. With log factor, maybe ~ 500k operations per check.
    # In Python, 500k operations might take 0.1-0.5 seconds.
    # So we can do maybe 100-500 checks in 60 seconds.
    
    # Let's do a local search: flip a random sign, if it improves, keep it.
    # To make it faster, we can only check the progressions affected by the flip.
    # But that's complex. Let's just do full checks if we can afford it.
    
    # Let's try to speed up the check.
    # The current check is O(n^2) because for each d, we process n elements total.
    # So total operations are sum_{d=1}^{n} n = n^2 = 65536.
    # Plus the Kadane's algorithm overhead.
    # This should be fast enough in NumPy or even pure Python if optimized.
    
    # Let's use NumPy for the check to be safe.
    
    def max_discrepancy_fast(signs_arr):
        n = len(signs_arr)
        max_abs = 0
        for d in range(1, n + 1):
            # Reshape or use slicing
            # For each r in 0..d-1
            for r in range(d):
                seq = signs_arr[r::d]
                if len(seq) == 0:
                    continue
                # Compute prefix sums
                prefix = np.concatenate(([0], np.cumsum(seq)))
                # We want max |prefix[j] - prefix[i]| for j > i
                # This is equivalent to max(prefix) - min(prefix) ? No, that's max range.
                # We need max difference between any two prefix sums where the later one is larger (for positive) or smaller (for negative).
                # Actually, max subarray sum is max(prefix[j] - min(prefix[0..j-1])).
                # And min subarray sum is min(prefix[j] - max(prefix[0..j-1])).
                
                # Let's compute running min and max.
                running_min = np.minimum.accumulate(prefix)
                running_max = np.maximum.accumulate(prefix)
                
                # For each j, the max sum ending at j is prefix[j] - running_min[j-1] (if j>0)
                # The min sum ending at j is prefix[j] - running_max[j-1]
                
                # We can vectorize this.
                # Note: prefix[0] = 0.
                # For j from 1 to len(prefix)-1:
                # max_end_j = prefix[j] - running_min[j-1]
                # min_end_j = prefix[j] - running_max[j-1]
                
                if len(prefix) > 1:
                    max_ends = prefix[1:] - running_min[:-1]
                    min_ends = prefix[1:] - running_max[:-1]
                    
                    max_pos = np.max(max_ends)
                    max_neg = np.min(min_ends)
                    
                    max_abs = max(max_abs, max_pos, -max_neg)
        
        return max_abs

    # Let's test the speed
    import time
    start = time.time()
    tm_disc_fast = max_discrepancy_fast(np.array(signs))
    end = time.time()
    # print(f"Time: {end-start}, Discrepancy: {tm_disc_fast}")
    
    # If it's too slow, we need to optimize further.
    # But let's assume it's okay.
    
    # Now, local search.
    # We'll try flipping signs.
    # To be efficient, we can use a simulated annealing approach or just hill climbing.
    # Let's do hill climbing with random restarts if stuck.
    
    best_signs = np.array(signs)
    best_disc = tm_disc_fast
    
    # We have about 60 seconds. Let's use 50 seconds for search.
    # We need to estimate how many evaluations we can do.
    # Let's do one evaluation to measure time.
    
    # Actually, let's just do a simple hill climbing loop until time runs out.
    
    import random
    rng = random.Random(42)
    
    current_signs = np.array(signs)
    current_disc = tm_disc_fast
    
    # We'll keep track of the best found.
    
    # To speed up, we can precompute some things? No, the structure changes with flips.
    
    # Let's just loop.
    
    # Note: max_discrepancy_fast is O(n^2) with small constants due to NumPy.
    # n=256, n^2=65536. NumPy operations on small arrays have overhead.
    # Let's see if we can do it in pure Python with optimizations.
    
    # Pure Python check might be slower per operation but no NumPy overhead.
    # Let's stick with NumPy for now and see.
    
    # Let's add a timeout mechanism.
    
    start_time = time.time()
    time_limit = 50.0  # Leave 10 seconds for safety
    
    while time.time() - start_time < time_limit:
        # Flip a random sign
        idx = rng.randint(0, n - 1)
        old_sign = current_signs[idx]
        current_signs[idx] = -old_sign
        
        new_disc = max_discrepancy_fast(current_signs)
        
        if new_disc < current_disc:
            current_disc = new_disc
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = current_signs.copy()
        else:
            # Revert
            current_signs[idx] = old_sign
            # Optional: if we've been stuck for a while, do a random restart or perturbation
            # For simplicity, just continue.
    
    return [int(x) for x in best_signs]

# EVOLVE-BLOCK-END
