import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # 1. Precompute primes up to sqrt(10^12) = 10^6
    LIMIT = 10**6
    is_prime = np.ones(LIMIT + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(LIMIT)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.nonzero(is_prime)[0]
    
    # 2. Segmented processing
    S = 2 * 10**6  # Segment size
    L = 1
    R = L + S - 1
    
    # Sliding window state
    counts = {}
    l = 1
    best_k = 1
    best_s = 1
    r_global = 1
    
    while R <= 10**12:
        # Check time
        if time.time() - start_time > deadline - 5:
            break
            
        # Compute phi for [L, R]
        # phi[n] = n initially
        phi_seg = np.arange(L, R + 1, dtype=np.int64)
        
        # Apply totient updates for each prime
        # For each prime p, find multiples in [L, R]
        for p in primes:
            if p > R:
                break
            # First multiple of p >= L
            start = ((L + p - 1) // p) * p
            if start > R:
                continue
            # Indices in phi_seg corresponding to multiples of p
            # start = L + idx => idx = start - L
            idx_start = start - L
            # Step is p
            # Update phi[n] = phi[n] / p * (p-1)
            # Vectorized update
            phi_seg[idx_start::p] = phi_seg[idx_start::p] // p * (p - 1)
            
        # Sliding window
        # Iterate through the segment
        for i in range(S):
            r_val = phi_seg[i]
            r_global = L + i
            
            # Add r_val to window
            if r_val in counts:
                # Shrink window from left until r_val is no longer in counts
                while l < r_global:
                    left_val = phi_seg[l - L] if l >= L else 0 # Need access to phi(l)
                    # Wait, l might be in a previous segment!
                    # We need to store phi values for the current window.
                    # The window size is small (k < 100).
                    # So we can maintain a deque or list of (index, phi_val) for the window.
                    pass
            
            # This approach fails if l < L.
            # We need to keep track of phi values for indices in [l, r_global].
            # Since k is small, we can store them in a dictionary or list.
            
            # Let's maintain a list `window_vals` of phi values in the current window [l, r_global].
            # And a dict `counts` of frequencies.
            
        # The above loop structure is problematic because we need phi(l) when l < L.
        # Solution: Keep a circular buffer or just a list of recent phi values.
        # Since k is small, we can just store the last S phi values? No, S is large.
        # We only need phi values for indices in [l, r_global].
        # l is close to r_global.
        # We can maintain a deque of (index, phi_value).
        
        # Let's restart the sliding window logic properly.
        # We process indices r_global from L to R.
        # We maintain `l` (left pointer of window).
        # We maintain `counts` dict.
        # We need to be able to get phi(l) when we increment l.
        # If l < L, we don't have phi(l) in `phi_seg`.
        # So we must have stored it previously.
        
        # Let's use a dictionary `phi_cache` for the current window?
        # Or simply a list `recent_phis` aligned with indices?
        # Since we process sequentially, we can just keep a list of phi values for the current segment.
        # And for previous segments, we only need the values that are still in the window.
        # The window size is `current_k`.
        # So we can maintain a deque `window_items` of `(index, phi_val)`.
        
        # Let's rewrite the loop.
        
        pass # Placeholder
        
        L = R + 1
        R = L + S - 1
        
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
