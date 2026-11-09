"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def time_ok():
        return elapsed() < deadline
    
    # Strategy: Use sieve to compute phi for a large range, then scan for longest run of distinct phi values.
    # The key insight is that phi values grow roughly linearly, so collisions are relatively rare
    # but do occur. We want to find the longest consecutive run where all phi values are distinct.
    
    # First, try a moderate range with sieve-based phi computation
    # We'll compute phi for range [1, N] using a sieve
    
    # Let's start with N = 200000 (200k) which should be fast enough
    N = 200000
    
    # Sieve for phi
    phi = list(range(N + 1))
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
    
    # Now scan for longest run of distinct phi values
    # For each starting position s, find how many consecutive values have distinct phi
    # We can do this efficiently with a sliding window approach
    
    best_k = 1
    best_s = 1
    
    # Sliding window: maintain a set of phi values in the current window
    # When we extend the window, check if phi[s+k] is already in the set
    # If yes, shrink from the left until it's not
    
    # Actually, for finding the longest run where ALL values are distinct,
    # we can use a two-pointer / sliding window technique
    
    seen = {}  # phi_value -> last index seen
    left = 1
    for right in range(1, N + 1):
        pv = phi[right]
        if pv in seen and seen[pv] >= left:
            # Collision: move left past the previous occurrence
            left = seen[pv] + 1
        seen[pv] = right
        current_len = right - left + 1
        if current_len > best_k:
            best_k = current_len
            best_s = left
        if not time_ok():
            break
    
    # Now try larger ranges if we have time
    # Try N = 1000000 (1M)
    if time_ok():
        N2 = 1000000
        phi2 = list(range(N2 + 1))
        for i in range(2, N2 + 1):
            if phi2[i] == i:
                for j in range(i, N2 + 1, i):
                    phi2[j] -= phi2[j] // i
        
        seen2 = {}
        left2 = 1
        best_k2 = 1
        best_s2 = 1
        for right in range(1, N2 + 1):
            pv = phi2[right]
            if pv in seen2 and seen2[pv] >= left2:
                left2 = seen2[pv] + 1
            seen2[pv] = right
            current_len = right - left2 + 1
            if current_len > best_k2:
                best_k2 = current_len
                best_s2 = left2
            if not time_ok():
                break
        
        if best_k2 > best_k:
            best_k = best_k2
            best_s = best_s2
    
    # Try even larger if time permits
    if time_ok():
        N3 = 5000000
        phi3 = list(range(N3 + 1))
        for i in range(2, N3 + 1):
            if phi3[i] == i:
                for j in range(i, N3 + 1, i):
                    phi3[j] -= phi3[j] // i
        
        seen3 = {}
        left3 = 1
        best_k3 = 1
        best_s3 = 1
        for right in range(1, N3 + 1):
            pv = phi3[right]
            if pv in seen3 and seen3[pv] >= left3:
                left3 = seen3[pv] + 1
            seen3[pv] = right
            current_len = right - left3 + 1
            if current_len > best_k3:
                best_k3 = current_len
                best_s3 = left3
            if not time_ok():
                break
        
        if best_k3 > best_k:
            best_k = best_k3
            best_s = best_s3
    
    # Verify the result: check that phi values in the run are truly distinct
    # and that s + k - 1 <= 10^12 (which is always true for our ranges)
    # Also ensure s >= 1 and 1 <= k <= 200
    
    if best_s < 1:
        best_s = 1
    if best_k < 1:
        best_k = 1
    if best_k > 200:
        best_k = 200
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
        if best_k < 1:
            best_s = 1
            best_k = 1
    
    return {'start': best_s, 'length': best_k}

# EVOLVE-BLOCK-END
