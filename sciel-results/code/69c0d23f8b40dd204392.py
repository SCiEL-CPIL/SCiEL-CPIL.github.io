"""Rewritten program: deterministic CRT construction + verification."""
import time
import math

def construct():
    deadline = time.time() + 130  # 130s safety margin within 150s limit
    
    def phi(x):
        """Compute Euler's totient of x exactly."""
        if x == 1:
            return 1
        result = x
        d = 2
        while d * d <= x:
            if x % d == 0:
                while x % d == 0:
                    x //= d
                result -= result // d
            d += 1
        if x > 1:
            result -= result // x
        return result
    
    # Strategy: Use CRT to find s such that phi(s), ..., phi(s+k-1) are distinct.
    # Key insight: if for each pair (i, j) with i < j in [0, k-1], we ensure
    # that s+i is NOT divisible by any prime p where p-1 | (j-i), then
    # phi(s+i) and phi(s+j) cannot be equal (since equal totients imply
    # specific divisibility relationships).
    #
    # More practically: search for good starting points using a sieve-like approach.
    # For each candidate s, compute phi(s), phi(s+1), ..., and check for distinctness.
    #
    # To make this efficient, we use the fact that phi values tend to be distinct
    # for "generic" numbers. We search in blocks and use early termination.
    #
    # Approach: 
    # 1. Try several candidate starting points in different ranges.
    # 2. For each candidate, extend the run as far as possible.
    # 3. Keep track of the best (longest) run found.
    #
    # For efficiency, we precompute phi for ranges using a sieve method.
    
    def compute_phi_sieve(n):
        """Compute phi for all numbers 1..n using sieve."""
        phi_arr = list(range(n + 1))
        for i in range(2, n + 1):
            if phi_arr[i] == i:  # i is prime
                for j in range(i, n + 1, i):
                    phi_arr[j] -= phi_arr[j] // i
        return phi_arr
    
    best_start = 1
    best_len = 1
    
    # Try different block sizes and starting points
    # The key is to find a long run of consecutive integers with distinct totients.
    
    # First, try a moderate range near 1 to get a baseline
    # Then try larger ranges
    
    # Block search strategy:
    # For a block [start, start + block_size), compute all phi values,
    # then find the longest sub-run with distinct values.
    
    def find_best_in_range(start, end, phi_arr, offset):
        """Find the longest run of distinct phi values in phi_arr[offset:start_offset:end_offset]."""
        # phi_arr is indexed from 0, and phi_arr[i] = phi(i+1) if we use 1-indexed
        # Actually, let's make it clear: phi_arr[x] = phi(x) for x in 1..n
        # We want to look at numbers start, start+1, ..., end
        # which correspond to phi_arr[start], phi_arr[start+1], ..., phi_arr[end]
        
        # Find longest run of distinct values
        best_s = start
        best_k = 1
        seen = set()
        k = 0
        for s in range(start, end + 1):
            # Reset and extend
            pass
        
        # More efficient: sliding window
        seen = set()
        left = start
        max_k = 0
        max_s = start
        for right in range(start, end + 1):
            val = phi_arr[right]
            while val in seen and left <= right:
                seen.remove(phi_arr[left])
                left += 1
            seen.add(val)
            if right - left + 1 > max_k:
                max_k = right - left + 1
                max_s = left
        return max_s, max_k
    
    # Try block size 50000 first
    block_size = 50000
    # Try starting at various points
    # We want to maximize k, so let's try to find runs of length up to 200
    
    # Start with a search in [1, 100000]
    search_end = min(100000, 10**12)
    phi_arr = compute_phi_sieve(search_end)
    
    # Find best run in this range
    s, k = find_best_in_range(1, search_end, phi_arr, 0)
    if k > best_len:
        best_len = k
        best_start = s
    
    if time.time() > deadline - 30:
        return {'start': best_start, 'length': best_len}
    
    # Try larger blocks
    # We need to be careful about memory and time
    # Let's try a block of size 200000 starting at different offsets
    
    block_size = 200000
    num_blocks = 5
    
    for block_idx in range(num_blocks):
        if time.time() > deadline - 20:
            break
        
        block_start = 1 + block_idx * block_size
        block_end = min(block_start + block_size - 1, 10**12)
        if block_start >= block_end:
            break
        
        # Compute phi for this block
        # We need phi for numbers block_start to block_end
        # This is block_size numbers
        phi_block = [0] * (block_size + 1)
        for i in range(1, block_size + 1):
            phi_block[i] = phi(block_start + i - 1)
        
        # Find best run in this block
        # phi_block[i] = phi(block_start + i - 1)
        # So number n corresponds to index n - block_start + 1
        best_s_local, best_k_local = 1, 1
        seen = set()
        left = 1
        max_k = 0
        max_s = 1
        for right in range(1, block_size + 1):
            val = phi_block[right]
            while val in seen and left <= right:
                seen.remove(phi_block[left])
                left += 1
            seen.add(val)
            if right - left + 1 > max_k:
                max_k = right - left + 1
                max_s = left
        
        if max_k > 0:
            actual_start = block_start + max_s - 1
            if max_k > best_len:
                best_len = max_k
                best_start = actual_start
    
    return {'start': best_start, 'length': best_len}
