# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 55.0  # safety margin before 60s kill

    # Precompute all arithmetic progressions
    # For each (start, step) pair, we store the list of indices
    # An AP is defined by start s, step d >= 1, length L >= 1
    # such that s + (L-1)*d < n
    # We want to minimize max over all APs of |sum_{k=0}^{L-1} x[s+k*d]|
    
    # Efficient approach: for each step d and start s, compute the prefix sums
    # along the progression, then the max absolute subarray sum (but we need
    # max absolute sum over ANY contiguous subsequence of the progression, 
    # i.e., any length from 1 to max_len starting at any position in the progression)
    
    # Actually, for a fixed (start, step), the progression is x[start], x[start+step], ...
    # Any sub-progression is a contiguous segment of this. So we need max over all 
    # contiguous segments of the absolute sum.
    
    # Let's precompute for each (step, start) the array of values along that AP.
    # Then compute the maximum absolute sum over all contiguous subarrays.
    
    # Number of APs: sum over step=1 to n-1, start=0 to step-1 of (number of valid starts)
    # For step d, start can be 0..d-1, and length is floor((n-1-start)/d)+1
    # Total number of (start, step) pairs is sum_{d=1}^{n-1} d = n*(n-1)/2 ≈ 32640 for n=256
    # Plus step=n? No, step must be < n for length >= 2. Step >= n gives only length 1.
    # Actually step can be up to n-1 for length >= 2. For length 1, every single element is an AP.
    # But length-1 APs just give |x[i]| = 1, so they don't affect the max (which is at least 1).
    
    # Let me think about this more carefully. The max absolute sum over any AP includes
    # singletons which are always 1. So the minimum possible max is at least 1.
    
    # For the optimization, we need to evaluate: for a given sign vector x, 
    # compute max over all (s, d, L) of |sum_{k=0}^{L-1} x[s+k*d]|
    
    # Efficient evaluation:
    # For each step d from 1 to n-1:
    #   For each start s from 0 to d-1:
    #     Compute prefix sums along the AP: P[0]=0, P[k+1] = P[k] + x[s+k*d]
    #     The max absolute sum over all contiguous segments = max over i<j of |P[j]-P[i]|
    #     = max(P) - min(P)
    
    # This is O(n^2) per step-start pair... no wait.
    # For fixed (d, s), the AP has length L = floor((n-1-s)/d)+1
    # Computing prefix sums is O(L), and finding max-min of prefix sums is O(L).
    # Total over all (d,s): sum_{d=1}^{n-1} sum_{s=0}^{d-1} L(s,d)
    # = sum_{d=1}^{n-1} sum_{s=0}^{d-1} floor((n-1-s)/d)+1
    # ≈ sum_{d=1}^{n-1} d * (n/d) = sum_{d=1}^{n-1} n = n*(n-1) ≈ 65536
    # That's very fast!
    
    def evaluate(x):
        """Compute max absolute sum over all arithmetic progressions."""
        max_abs = 1  # singletons give 1
        for d in range(1, n):
            for s in range(d):
                # Get indices: s, s+d, s+2d, ... < n
                # Compute prefix sums
                # Use numpy for speed
                idx = np.arange(s, n, d)
                vals = x[idx]
                # Prefix sum with 0 at start
                P = np.concatenate(([0], np.cumsum(vals)))
                # Max absolute sum = max(P) - min(P)
                local_max = int(P.max() - P.min())
                if local_max > max_abs:
                    max_abs = local_max
        return max_abs
    
    # Local search: flip one sign at a time
    def local_search(x, max_iters=100000):
        """Iteratively flip signs to reduce the max absolute AP sum."""
        x = x.copy()
        current_max = evaluate(x)
        best_x = x.copy()
        best_max = current_max
        
        rng = random.Random(42)
        no_improve = 0
        max_no_improve = 5000
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            if no_improve > max_no_improve:
                break
            
            # Pick a random position to flip
            i = rng.randint(0, n-1)
            old_val = x[i]
            x[i] = -old_val
            
            new_max = evaluate(x)
            if new_max < current_max:
                current_max = new_max
                no_improve = 0
                if current_max < best_max:
                    best_max = current_max
                    best_x = x.copy()
            else:
                # Revert
                x[i] = old_val
                no_improve += 1
        
        return best_x, best_max
    
    # Multi-start local search
    best_overall = None
    best_overall_max = 999
    
    # Start with a few random initializations
    rng = random.Random(171)
    
    # First, try a structured initialization based on Thue-Morse or similar
    # Thue-Morse sequence: t(n) = popcount(n) mod 2, mapped to ±1
    # This has good discrepancy properties for initial segments
    
    # Thue-Morse based initialization
    def thue_morse(n):
        """Generate Thue-Morse ±1 sequence of length n."""
        seq = [1]
        while len(seq) < n:
            seq = seq + [-s for s in seq]
        return seq[:n]
    
    # Try Thue-Morse
    x_tm = np.array(thue_morse(n), dtype=np.int8)
    x_tm, tm_max = local_search(x_tm, max_iters=50000)
    if tm_max < best_overall_max:
        best_overall_max = tm_max
        best_overall = x_tm.copy()
    
    if time.time() > deadline - 10:
        return [int(v) for v in best_overall]
    
    # Try a few random starts
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        x_rand = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        x_rand, rand_max = local_search(x_rand, max_iters=50000)
        if rand_max < best_overall_max:
            best_overall_max = rand_max
            best_overall = x_rand.copy()
    
    # Try one more structured approach: use a low-discrepancy-like pattern
    # Alternating blocks
    x_alt = np.array([(-1)**(i//4) for i in range(n)], dtype=np.int8)
    x_alt, alt_max = local_search(x_alt, max_iters=50000)
    if alt_max < best_overall_max:
        best_overall_max = alt_max
        best_overall = x_alt.copy()
    
    if best_overall is None:
        best_overall = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
    
    return [int(v) for v in best_overall]

# EVOLVE-BLOCK-END
