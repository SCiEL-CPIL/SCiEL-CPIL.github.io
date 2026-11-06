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
    
    def find_longest_distinct_run(phi_vals):
        """
        Find the longest contiguous subsequence with distinct values.
        Returns (max_length, start_index).
        """
        if len(phi_vals) < 2:
            return 1, 1 if len(phi_vals) > 1 else 0
            
        last_seen = {}
        start = 1
        max_len = 0
        max_start = 1
        
        # phi_vals is 0-indexed, but we care about indices 1..N
        # phi_vals[0] is phi(0) which is 0, usually ignored.
        # The problem asks for s >= 1.
        
        for i in range(1, len(phi_vals)):
            val = phi_vals[i]
            if val in last_seen and last_seen[val] >= start:
                start = last_seen[val] + 1
            last_seen[val] = i
            
            current_len = i - start + 1
            if current_len > max_len:
                max_len = current_len
                max_start = start
                
        return max_len, max_start

    best_k = 1
    best_s = 1
    
    # Try increasing sieve limits
    # NumPy is fast, so we can try larger limits.
    # 10^7 is safe. 10^8 might be tight but possible.
    sieve_limits = [1000000, 5000000, 10000000, 50000000]
    
    for limit in sieve_limits:
        if time_remaining() < 10.0:
            break
            
        try:
            phi_vals = compute_phi_sieve_numpy(limit)
            k, s = find_longest_distinct_run(phi_vals)
            
            # Cap k at 200
            if k > 200:
                k = 200
                # We can start anywhere in the run of length > 200.
                # The run found is [max_start, max_start + original_k - 1]
                # We just need a valid start s such that s+k-1 <= limit.
                # Since original_k > 200, we can set s = max_start.
                # The run from max_start to max_start+199 is valid.
            else:
                s = s # s is already the start of the max run
                
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            # If memory or time issues, break
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
