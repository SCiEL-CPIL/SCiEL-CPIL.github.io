import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s wall-clock limit

    # Step 1: Sieve of Eratosthenes to generate primes up to 10^6
    # This allows trial division up to sqrt(10^12) = 10^6
    limit = 1000000
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def phi(x):
        """Compute Euler's totient phi(x) using trial division by primes up to sqrt(x)."""
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                while y % p == 0:
                    y //= p
                result -= result // p
        if y > 1:
            result -= result // y
        return result

    def phi_segment(start, length):
        """Compute phi values for a contiguous segment [start, start+length-1]."""
        return [phi(start + i) for i in range(length)]

    # Step 2: Search for long runs of consecutive integers with distinct totients
    # Strategy: 
    # 1. First, do a coarse scan over small ranges to find good starting points
    # 2. Then, expand promising regions with larger segments
    
    best_k = 1
    best_s = 1

    # Precompute a moderately sized segment to find initial candidates
    # Start with a segment of size ~5000 to get initial best
    seg_size = 5000
    seg_start = 1
    
    # We'll search in windows. For each window, compute phi values and find max distinct run.
    # To be efficient, we'll use a sliding window approach.
    
    # First, let's do a quick scan over smaller segments to find where long runs might be
    # Then focus on those regions
    
    # Approach: 
    # - Compute phi for a large range (e.g., 1 to 200000) in chunks
    # - Track the longest run of distinct phi values seen so far
    # - Use a set-based sliding window
    
    # Let's search in chunks to keep memory reasonable
    chunk_size = 100000
    
    # We need to track phi values across chunk boundaries for the sliding window
    # So we'll maintain a buffer of recent phi values
    
    buffer_size = 2000  # enough to cover any run up to 200 with margin
    phi_buffer = []
    buf_start = 1  # the integer value corresponding to phi_buffer[0]
    
    def extend_buffer(target_end):
        """Extend phi_buffer so it covers up to target_end (inclusive)."""
        nonlocal buf_start, phi_buffer
        while buf_start + len(phi_buffer) <= target_end:
            # Add next value
            next_val = buf_start + len(phi_buffer)
            phi_buffer.append(phi(next_val))
    
    # We'll scan from 1 upward, maintaining a window of distinct phi values
    # For efficiency, we'll compute in chunks and track the best run
    
    # Let's do a two-phase search:
    # Phase 1: Quick scan up to ~200k to find candidate regions
    # Phase 2: Focused search around promising areas
    
    # Actually, let's just do a straightforward scan with a sliding window
    # The key insight: if we have a run of length k starting at s, then 
    # phi(s), phi(s+1), ..., phi(s+k-1) are all distinct.
    
    # We'll maintain:
    # - A list of phi values for the current window
    # - A set of phi values in the current window
    # - The left pointer of the window
    
    # To handle the constraint that we need exact factorization, we use our phi() function.
    
    # Let's scan up to a reasonable limit, checking time periodically
    max_scan = 500000  # scan up to 500k integers
    
    # Use a sliding window with two pointers
    left = 1
    current_window = {}  # phi_value -> last index where it appeared
    current_count = 0
    best_in_window = 0
    best_start_in_window = 1
    
    # We'll compute phi values one by one and maintain the sliding window
    # But computing phi one by one might be slow for large ranges.
    
    # Alternative: compute phi in batches, then process
    
    # Let's use a batch approach
    batch_size = 5000
    s = 1
    while s <= max_scan and time.time() < deadline - 5:
        # Compute phi for a batch
        batch_end = min(s + batch_size - 1, max_scan)
        batch_len = batch_end - s + 1
        phi_vals = [phi(s + i) for i in range(batch_len)]
        
        # Now process this batch with the sliding window
        # We need to maintain state across batches
        # Let's use a simpler approach: for each starting position in the batch,
        # extend the run as far as possible within the batch and beyond
        
        # Actually, let's use a different strategy:
        # For each position s in the batch, try to extend the run of distinct phi values
        # starting from s
        
        # But this could be O(n^2) in the worst case.
        
        # Better: maintain a global sliding window
        # We'll track for each phi value, the most recent position where it was seen
        # If we see a duplicate, we need to move the left pointer past the previous occurrence
        
        # Let's maintain:
        # last_occurrence[phi_val] = most recent index where phi_val was seen
        # left = left pointer of the current valid window
        
        # We'll process each element in the batch
        for i, pv in enumerate(phi_vals):
            idx = s + i  # the integer value
            
            if pv in current_window:
                # Move left pointer past the previous occurrence
                while left <= current_window[pv]:
                    # Remove phi value at position left
                    lv = phi_vals[0] if left == s else None
                    # Actually, we don't have all previous phi values stored
                    # We need a different approach
                    break
                # This approach is getting complicated. Let's simplify.
            
            # Let's use a simpler but potentially slower approach for now:
            # For each starting position, count the length of the distinct run
            # But optimize by stopping early if we can't beat the current best
        
            # Actually, let's just do a direct search for each starting position
            # but limit the maximum run length we check to best_k + some margin
            
            # Reset and do a cleaner implementation
            break
        
        # Let's restart with a cleaner approach
        break
    
    # Clean implementation:
    # For each starting position s, compute the length of the longest run of distinct phi values
    # starting from s. Use early termination if we can't beat the current best.
    
    best_k = 1
    best_s = 1
    
    # We'll search from 1 to max_scan
    # For efficiency, we'll compute phi values and cache them
    
    # Let's use a different strategy: 
    # 1. Compute phi values for a range
    # 2. Use a sliding window to find the longest subarray with all distinct values
    
    # The sliding window approach:
    # - left and right pointers
    # - A set (or dict) to track phi values in the current window
    # - When we see a duplicate, move left until the duplicate is removed
    
    # Let's implement this properly
    
    max_scan = 300000  # scan up to 300k integers
    # Compute phi values for the entire range
    phi_vals = [0] * (max_scan + 1)  # 1-indexed
    phi_vals[0] = 0
    
    # Compute phi values in a loop
    for i in range(1, max_scan + 1):
        if time.time() > deadline - 10:
            break
        phi_vals[i] = phi(i)
    
    # Now find the longest run of distinct phi values
    # Using sliding window
    left = 1
    seen = {}  # phi_value -> count in current window
    current_max = 0
    current_start = 1
    
    right = 1
    for right in range(1, max_scan + 1):
        pv = phi_vals[right]
        if pv in seen:
            seen[pv] += 1
        else:
            seen[pv] = 1
        
        # Shrink window from the left if there are duplicates
        while len(seen) < right - left + 1:
            left_pv = phi_vals[left]
            seen[left_pv] -= 1
            if seen[left_pv] == 0:
                del seen[left_pv]
            left += 1
        
        # Now the window [left, right] has all distinct phi values
        window_len = right - left + 1
        if window_len > current_max:
            current_max = window_len
            current_start = left
        
        if time.time() > deadline - 5:
            break
    
    if current_max > best_k:
        best_k = current_max
        best_s = current_start
    
    # If we have time, try to search beyond max_scan
    # Focus on regions near the best found so far
    
    if time.time() < deadline - 10 and best_k < 200:
        # Try searching in a wider range
        extended_max = 500000
        # We already have phi_vals up to max_scan
        # Extend the search
        
        # Actually, let's just return what we have for now
        # The sliding window should have found a good answer
    
    return {'start': best_s, 'length': best_k}
