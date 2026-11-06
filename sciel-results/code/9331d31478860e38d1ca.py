"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    def compute_phi_sieve_numpy(limit):
        """Compute phi for all numbers up to limit using NumPy."""
        phi = np.arange(limit + 1, dtype=np.int64)
        for i in range(2, limit + 1):
            if phi[i] == i:  # i is prime
                # Vectorized update
                indices = np.arange(i, limit + 1, i)
                phi[indices] -= phi[indices] // i
        return phi
    
    def find_longest_distinct_run_sliding(phi_vals, start_offset=1):
        """
        Find the longest contiguous subsequence with distinct values using sliding window.
        phi_vals: array where phi_vals[i] = phi(start_offset + i)
        Returns (max_length, start_index_in_phi_vals)
        """
        n = len(phi_vals)
        if n < 2:
            return 1, 0
            
        last_seen = {}
        left = 0
        max_len = 0
        max_start = 0
        
        for right in range(n):
            val = phi_vals[right]
            if val in last_seen and last_seen[val] >= left:
                left = last_seen[val] + 1
            last_seen[val] = right
            
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
                max_start = left
                
        return max_len, max_start

    best_k = 1
    best_s = 1
    
    # Try increasing sieve limits
    sieve_limits = [1000000, 5000000, 10000000]
    
    for limit in sieve_limits:
        if time_remaining() < 15.0:
            break
            
        try:
            phi_vals = compute_phi_sieve_numpy(limit)
            # phi_vals[0] = phi(0), phi_vals[1] = phi(1), etc.
            # We care about indices 1..limit
            k, rel_start = find_longest_distinct_run_sliding(phi_vals[1:])
            s = rel_start + 1  # Convert to actual number
            
            if k > 200:
                k = 200
            
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            break
            
    # Ensure valid return
    if best_k < 1:
        best_k = 1
        best_s = 1
        
    # Final safety cap
    if best_k > 200:
        best_k = 200
        
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
