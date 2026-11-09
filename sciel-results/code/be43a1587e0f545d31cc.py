"""Rewritten program for low discrepancy on all arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 52.0  # safety margin before 60s kill

    # Precompute all arithmetic progressions as index arrays (0-indexed)
    # For each step d >= 1 and start s (0-indexed), the AP is s, s+d, s+2d, ...
    # We need to consider ALL contiguous subsequences of each AP.
    # For a fixed (start, step), the AP gives a sequence of indices.
    # The max |sum| over any contiguous subsequence of that sequence is what matters.
    #
    # To efficiently evaluate: for each (start, step), compute prefix sums of the
    # signs along that AP, then max|sum| = max(prefix) - min(prefix) over that AP.
    # But we also need to consider single elements (length 1 APs), which are just the signs themselves.
    # Actually, the max absolute sum over any contiguous subsequence of the AP sequence
    # is max(prefix) - min(prefix), where prefix includes 0 at the start.
    
    # Let's precompute all APs as lists of indices
    aps = []
    for step in range(1, n + 1):
        for start in range(n):
            # AP: start, start+step, start+2*step, ... < n
            indices = list(range(start, n, step))
            if len(indices) >= 1:
                aps.append(np.array(indices, dtype=np.int32))
    
    # We have many APs. For each AP, we need to compute the max absolute contiguous sum.
    # This is equivalent to: given the sequence of signs along the AP, compute
    # max_subarray_sum_abs = max(max_subarray_sum, -min_subarray_sum)
    # where max_subarray_sum is the maximum sum of any contiguous subarray.
    # Using Kadane's algorithm variant.
    
    # For efficiency, let's think about the structure.
    # Number of APs: for step=1, there are 256 APs (starts 0..255), each of varying length.
    # For step=2, there are 256 APs.
    # Total APs = sum over step=1..256 of 256 = 256*256 = 65536 APs.
    # Each AP has length at most n/step + 1.
    # Total elements across all APs = sum_{step=1}^{256} sum_{start=0}^{255} (number of terms in AP starting at start with step step)
    # = sum_{step=1}^{256} sum_{start=0}^{255} floor((255-start)/step) + 1
    # This is roughly n * sum_{step=1}^{n} 1/step * n ≈ n^2 * H(n) ≈ 256^2 * 6 ≈ 393k elements.
    # That's manageable.
    
    # Let's build a more efficient representation.
    # For each (start, step), the AP indices are start, start+step, start+2*step, ...
    # We can compute the prefix sums along each AP.
    
    # Strategy: 
    # 1. Start with a good initial sequence (e.g., Thue-Morse or random)
    # 2. Use local search: flip individual signs to reduce the max discrepancy
    
    # Let's first compute the discrepancy for a given sign array.
    def compute_max_discrepancy(signs):
        """Compute max absolute sum over any arithmetic progression."""
        signs = np.array(signs, dtype=np.int8)
        max_disc = 0
        for step in range(1, n + 1):
            for start in range(n):
                # Get the AP indices
                idx = np.arange(start, n, step, dtype=np.int32)
                if len(idx) == 0:
                    continue
                # Get signs along this AP
                s = signs[idx]
                # Compute prefix sums
                prefix = np.concatenate(([0], np.cumsum(s)))
                # Max absolute contiguous sum = max(prefix) - min(prefix)
                disc = int(prefix.max() - prefix.min())
                if disc > max_disc:
                    max_disc = disc
        return max_disc
    
    # Initial sequence: use Thue-Morse sequence
    def thue_morse(n):
        """Generate Thue-Morse sequence of length n, values -1 or +1."""
        seq = [1]
        for i in range(n - 1):
            seq.append(-seq[i])
        return seq[:n]
    
    # Start with Thue-Morse
    signs = thue_morse(n)
    current_disc = compute_max_discrepancy(signs)
    best_signs = signs[:]
    best_disc = current_disc
    
    # Local search: try flipping each position and see if it reduces the discrepancy
    # To speed up, we can use the fact that flipping position i only affects APs that contain i.
    # But for now, let's do a simpler approach with time budget.
    
    rng = random.Random(42)
    improvement_count = 0
    no_improve_steps = 0
    max_no_improve = 500
    
    while time.time() < deadline - 5.0:
        # Pick a random position to flip
        pos = rng.randint(0, n - 1)
        
        # Flip it
        signs[pos] = -signs[pos]
        
        # Compute new discrepancy
        new_disc = compute_max_discrepancy(signs)
        
        if new_disc < current_disc:
            current_disc = new_disc
            improvement_count += 1
            no_improve_steps = 0
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs[:]
        else:
            # Revert
            signs[pos] = -signs[pos]
            no_improve_steps += 1
        
        # If we've had too many non-improving steps, restart with a different initial sequence
        if no_improve_steps > max_no_improve:
            # Try a different random initial sequence
            signs = [rng.choice((-1, 1)) for _ in range(n)]
            current_disc = compute_max_discrepancy(signs)
            no_improve_steps = 0
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs[:]
    
    return best_signs
# EVOLVE-BLOCK-END
