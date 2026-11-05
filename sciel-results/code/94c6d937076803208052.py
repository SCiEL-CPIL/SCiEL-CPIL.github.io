import math
import time
import random

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, aim for 130s usage
    DEADLINE = start_time + 130.0
    
    # 1. Precompute primes up to 10^6
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    if limit_p >= 1:
        is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    best_k = 1
    best_s = 1
    
    # Window size for segmented sieve
    # 10^6 is a good balance between memory and CPU efficiency
    W = 10**6 
    max_range = 10**12
    
    def find_longest_distinct_run(phi_arr, W):
        """
        Finds the longest subarray in phi_arr with all distinct elements.
        Returns (length, start_index_in_arr)
        """
        seen = {}
        max_len = 0
        max_start = 0
        left = 0
        
        for right in range(W):
            val = phi_arr[right]
            if val in seen and seen[val] >= left:
                # Collision: move left pointer to just after the previous occurrence
                left = seen[val] + 1
            
            seen[val] = right
            
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
                max_start = left
        
        return max_len, max_start

    def process_window(L, W):
        """
        Computes phi for [L, L+W) using segmented sieve and finds longest distinct run.
        Returns (k, s)
        """
        # Initialize phi array
        # Using list of ints
        phi_arr = [L + i for i in range(W)]
        
        # Sieve
        for p in primes:
            if p * p > L + W - 1:
                # Optimization: if p*p > max_val, we still need to handle p if p is in range?
                # Actually, the standard segmented sieve logic works for all primes.
                # If p > sqrt(max_val), each number in the range can have at most one factor p.
                # The loop below handles this correctly.
                pass
            
            # Find first multiple of p in [L, L+W)
            start = ((L + p - 1) // p) * p
            
            # If start < p, it means L was 0 or small, but L >= 1.
            # If start == p, and p is in range, we should NOT divide p by p in the phi calc?
            # phi(p) = p - 1.
            # Formula: phi(n) = n * (1 - 1/p).
            # If n=p, phi(p) = p - p/p = p-1. Correct.
            # So we do include p itself.
            
            for j in range(start, L + W, p):
                idx = j - L
                phi_arr[idx] -= phi_arr[idx] // p
        
        # Find longest run
        k, idx_start = find_longest_distinct_run(phi_arr, W)
        s = L + idx_start
        
        return k, s

    # Search Strategy
    # We will scan random windows.
    # To avoid overlapping too much, we can keep a set of visited L's, but memory is cheap for ints.
    # Or just random.
    
    # Initial check with small numbers to ensure baseline
    # process_window(1, 1000)
    
    # Main loop
    while time.time() < DEADLINE - 2.0:
        # Choose random start L
        # Ensure L + W <= 10^12
        L = random.randint(1, max_range - W)
        
        # Check if we have time to do this window
        # Estimate time: ~50ms per window? 
        # If we are close to deadline, skip large windows or break.
        
        k, s = process_window(L, W)
        
        if k > best_k:
            best_k = k
            best_s = s
            
        if best_k == 200:
            break
            
        # Optional: If we find a very good k, we might refine around it?
        # For now, just continue sampling.
        
        # Check time
        if time.time() > DEADLINE - 5.0:
            break

    return {'start': best_s, 'length': best_k}
