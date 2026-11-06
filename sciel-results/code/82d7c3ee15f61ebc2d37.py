import itertools
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    N = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, N + 1):
        L = L * i // gcd(L, i)
        
    # Precompute weights w_i = L // i
    weights = [0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = L // i

    best_abs_num = float('inf')
    best_coeffs = [0] * N

    # We search for the minimum non-zero absolute numerator
    # Start with k=2, then k=3, etc.
    # The number of candidates grows rapidly:
    # k=2: ~1.7k
    # k=3: ~32k
    # k=4: ~438k
    # k=5: ~4.5M
    # k=6: ~38M (might be too slow in Python)
    
    max_k = 6
    
    for k in range(2, max_k + 1):
        if time.time() > deadline - 5:
            break
            
        # Iterate over all combinations of k indices
        # indices are 1-based in weights, but list is 0-based
        # We use indices 0..N-1 corresponding to i=1..N
        indices = range(1, N + 1)
        
        for combo in itertools.combinations(indices, k):
            # For this combination, try all 2^k sign assignments
            # We can represent signs as a bitmask
            # But iterating 2^k for each combo might be slow if done naively in a loop
            # Better: generate sign vectors recursively or use bit manipulation
            
            # Optimization: 
            # Let S = sum s_i * w_i.
            # We can split the combo into two halves and use meet-in-the-middle?
            # For k<=6, simple enumeration is fine.
            
            # To speed up, we can precompute the base sum for all +1, 
            # then flip signs.
            # Sum = sum(w_i) - 2 * sum(w_j for j where s_j = -1)
            
            total_pos = 0
            for idx in combo:
                total_pos += weights[idx]
                
            # Now we need to find a subset of combo such that 
            # |total_pos - 2 * subset_sum| is minimized and non-zero.
            # subset_sum is the sum of weights for indices assigned -1.
            
            # Number of subsets is 2^k.
            # We can iterate through all masks.
            for mask in range(1 << k):
                # Calculate subset sum for this mask
                # mask bit i is 1 if combo[i] is assigned -1
                subset_sum = 0
                # Unrolling for small k might help, but loop is okay for k<=6
                for j in range(k):
                    if (mask >> j) & 1:
                        subset_sum += weights[combo[j]]
                
                num = total_pos - 2 * subset_sum
                abs_num = abs(num)
                
                if abs_num > 0 and abs_num < best_abs_num:
                    best_abs_num = abs_num
                    # Construct coefficients
                    coeffs = [0] * N
                    for j in range(k):
                        idx = combo[j]
                        if (mask >> j) & 1:
                            coeffs[idx - 1] = -1
                        else:
                            coeffs[idx - 1] = 1
                    best_coeffs = coeffs
                    
                    # Optional: Early termination if we hit 1?
                    # The numerator is an integer. The smallest non-zero integer is 1.
                    # If we find abs_num == 1, it's the global minimum possible.
                    if best_abs_num == 1:
                        return best_coeffs

    # If no non-zero sum found (should not happen), return a default valid non-zero sum
    # e.g., c_1 = 1, others 0 -> sum = 1
    if best_abs_num == float('inf'):
        best_coeffs = [1] + [0] * (N - 1)
        
    return best_coeffs
