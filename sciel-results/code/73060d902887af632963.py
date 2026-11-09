"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def _sigma_sieve(n, deadline=None):
    """Compute sigma(k) for k=1..n using a numpy-accelerated sieve."""
    sigma = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        if deadline is not None and time.time() > deadline:
            break
        sigma[i::i] += i
    return sigma

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Start with known good result
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Try progressively larger sieve sizes
    # The key insight: larger n gives more chances for k*sigma(k) collisions
    sieve_sizes = [500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000]
    
    for n in sieve_sizes:
        elapsed = time.time() - start_time
        if elapsed > deadline - 35:
            break
        
        # Estimate time for this sieve: roughly O(n log n)
        # For n=100M, this takes ~30-60s in numpy
        if n > 10000000 and elapsed > 50:
            break
        if n > 20000000 and elapsed > 80:
            break
        if n > 50000000 and elapsed > 100:
            break
        if n > 100000000 and elapsed > 115:
            break
        
        sigma = _sigma_sieve(n, deadline=deadline - 25)
        
        if time.time() - start_time > deadline - 20:
            break
        
        # Group k by M = k * sigma(k) using efficient approach
        # Use a dictionary to count occurrences
        groups = defaultdict(list)
        
        # Vectorized computation of M values
        ks_array = np.arange(1, n + 1, dtype=np.int64)
        M_values = ks_array * sigma[1:]
        
        # Filter to M <= 10^18
        mask = M_values <= 10**18
        valid_M = M_values[mask]
        valid_k = ks_array[mask]
        
        # Convert to Python ints for dictionary use
        # Process in chunks to avoid memory issues
        chunk_size = 1000000
        for start in range(0, len(valid_M), chunk_size):
            end = min(start + chunk_size, len(valid_M))
            for i in range(start, end):
                m = int(valid_M[i])
                k = int(valid_k[i])
                groups[m].append(k)
        
        # Find M with most solutions
        for M, ks in groups.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_M = M
                best_ks = ks[:1000]
        
        # Clear groups to free memory
        del groups
        del sigma
        del M_values
        del valid_M
        del valid_k
    
    # Verify the result
    def _sigma_verify(k):
        """Compute sigma(k) by trial division."""
        if k == 1:
            return 1
        result = 1
        temp = k
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                result *= (p**(exp+1) - 1) // (p - 1)
            p += 1 if p == 2 else 2
        if temp > 1:
            result *= (temp**2 - 1) // (temp - 1)
        return result
    
    # Verify all k values
    verified_ks = []
    for k in best_ks:
        if k * _sigma_verify(k) == best_M and k <= math.isqrt(best_M):
            verified_ks.append(k)
    
    if len(verified_ks) >= 1:
        return {'M': int(best_M), 'k': [int(k) for k in verified_ks]}
    else:
        return {'M': 333312, 'k': [336, 372, 434]}
# EVOLVE-BLOCK-END
