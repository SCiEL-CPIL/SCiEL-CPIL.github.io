import numpy as np
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = 55.0

    # Precompute largest prime factor for all numbers up to 1000001
    max_val = 1000001
    size = max_val + 1
    lpf = np.zeros(size, dtype=np.int32)
    for i in range(2, size):
        if lpf[i] == 0:
            lpf[i::i] = i

    if time.time() - start_time > deadline:
        return 10000

    # Full scan to find the global minimum
    lpf_n = lpf[10000:1000001]
    lpf_n1 = lpf[10001:1000002]
    max_lpf_pairs = np.maximum(lpf_n, lpf_n1)
    min_val = int(np.min(max_lpf_pairs))
    best_n = int(10000 + np.argmin(max_lpf_pairs))

    if time.time() - start_time > deadline:
        return best_n

    # Now try to find a better solution by exploring the structure
    # We want both n and n+1 to have small largest prime factors
    # This means both should be "smooth" numbers
    
    # Strategy: Generate smooth numbers (numbers with only small prime factors)
    # and check if both n and n+1 are smooth
    
    # Let's define "smooth" as having all prime factors <= some bound
    # We'll try different smoothness bounds and see what we find
    
    # First, let's see what the current best value is
    # min_val is the largest prime factor of n(n+1) for the best n found
    
    # Now, let's try to find pairs where both n and n+1 are very smooth
    # A number is y-smooth if all its prime factors are <= y
    
    # Generate all y-smooth numbers up to 1000000 for various y
    # and check if consecutive pairs exist
    
    best_score = min_val
    best_candidate = best_n
    
    # Try different smoothness bounds
    for y in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if time.time() - start_time > deadline:
            break
        
        # Generate all y-smooth numbers up to 1000000
        # A y-smooth number has only prime factors <= y
        smooth_numbers = set()
        
        # BFS to generate smooth numbers
        queue = deque([1])
        primes = [p for p in range(2, y + 1) if lpf[p] == p]  # primes <= y
        
        while queue:
            x = queue.popleft()
            if x > 1000000:
                continue
            smooth_numbers.add(x)
            for p in primes:
                if x * p <= 1000000:
                    queue.append(x * p)
        
        # Convert to sorted list for efficient checking
        smooth_list = sorted(smooth_numbers)
        
        # Check for consecutive pairs in the smooth numbers
        # We want n and n+1 both in smooth_numbers
        smooth_set = set(smooth_list)
        
        for n in smooth_list:
            if n < 10000 or n + 1 > 1000000:
                continue
            if (n + 1) in smooth_set:
                # Both n and n+1 are y-smooth
                # The largest prime factor of n(n+1) is <= y
                # So the score is at most y
                if y < best_score:
                    best_score = y
                    best_candidate = n
        
        # Also check if n is y-smooth and n+1 has a small LPF, or vice versa
        # More generally, we want max(LPF(n), LPF(n+1)) to be small
        # Let's check all pairs where one is smooth and the other has small LPF
        
    # Additionally, let's do a targeted search around the current best
    # Look for n where both n and n+1 have LPF <= some threshold
    threshold = best_score
    for n in range(10000, min(1000000, 10000 + 200000)):
        if time.time() - start_time > deadline:
            break
        max_lpf = max(int(lpf[n]), int(lpf[n + 1]))
        if max_lpf < best_score:
            best_score = max_lpf
            best_candidate = n
    
    # Final check: scan the entire range again with the improved threshold
    # to make sure we haven't missed anything
    if time.time() - start_time < deadline - 5:
        for n in range(10000, 1000000):
            max_lpf = max(int(lpf[n]), int(lpf[n + 1]))
            if max_lpf < best_score:
                best_score = max_lpf
                best_candidate = n
            if time.time() - start_time > deadline:
                break
    
    return int(best_candidate)

