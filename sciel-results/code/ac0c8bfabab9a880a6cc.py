"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Use a moderate range to allow more search within time
    # The original used 20001, which is quite small
    # Let's try a larger range, but we need to be careful about phi computation time
    # We'll use a sieve-based phi computation for efficiency
    
    # First, let's try to find good runs in a larger range
    # We'll compute phi for a range and look for long runs of distinct values
    
    # Strategy: 
    # 1. Compute phi for a large range using sieve
    # 2. Scan for longest run of distinct phi values
    # 3. Use deadline guard to return best found
    
    # Let's start with a reasonable range size
    # phi computation via sieve is O(n log log n)
    # For n = 10^6, this should be fast
    
    # But we need s + k - 1 <= 10^12, so s can be large
    # However, finding long runs requires searching
    
    # Let's try multiple ranges of different sizes
    # Start with smaller ranges to get quick results, then expand if time permits
    
    best_k = 1
    best_s = 1
    
    # Try several window sizes and starting positions
    # We'll use a segmented approach
    
    # First, let's compute phi for a base range
    # We'll use a sieve to compute phi efficiently
    
    def compute_phi_sieve(n):
        """Compute phi for all numbers from 1 to n using sieve."""
        phi = list(range(n + 1))
        for i in range(2, n + 1):
            if phi[i] == i:  # i is prime
                for j in range(i, n + 1, i):
                    phi[j] -= phi[j] // i
        return phi
    
    # Try different range sizes
    # Start with 100000, then try larger if time permits
    range_sizes = [100000, 200000, 500000, 1000000]
    
    for range_size in range_sizes:
        if time.time() - start_time > deadline - 5:
            break
        
        # Compute phi for this range
        phi_vals = compute_phi_sieve(range_size)
        
        # Scan for longest run of distinct phi values
        for s in range(1, range_size - 200):
            if time.time() - start_time > deadline - 3:
                break
            
            seen = set()
            k = 0
            while s + k <= range_size and phi_vals[s + k] not in seen:
                seen.add(phi_vals[s + k])
                k += 1
            
            if k > best_k:
                best_k = k
                best_s = s
                # Early termination if we hit max possible
                if best_k >= 200:
                    return {'start': best_s, 'length': best_k}
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
