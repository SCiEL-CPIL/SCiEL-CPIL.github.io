import math
import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Fallback known solution
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Try progressively larger sieve sizes
    # With NumPy, we can handle much larger N
    sizes = [5_000_000, 20_000_000, 50_000_000, 100_000_000, 200_000_000, 500_000_000]
    
    for K_MAX in sizes:
        if time.time() > deadline - 30:
            break
            
        try:
            # Build SPF using NumPy
            spf = np.arange(K_MAX + 1, dtype=np.int64)
            if K_MAX >= 1:
                spf[0] = 0
            if K_MAX >= 1:
                spf[1] = 1
                
            # Sieve for SPF
            limit = int(math.isqrt(K_MAX))
            for i in range(2, limit + 1):
                if spf[i] == i:  # i is prime
                    start = i * i
                    step = i
                    # Vectorized: set spf[j] = i for j in [start, K_MAX] step i where spf[j] == j
                    indices = np.arange(start, K_MAX + 1, step)
                    mask = spf[indices] == indices
                    if np.any(mask):
                        spf[indices[mask]] = i
            
            # Compute sigma using multiplicative property
            # sigma[1] = 1
            sigma = np.zeros(K_MAX + 1, dtype=np.int64)
            sigma[1] = 1
            
            # For each k from 2 to K_MAX:
            # p = spf[k], find m = k with all factors of p removed, a = exponent
            # sigma[k] = sigma[m] * (1 + p + p^2 + ... + p^a)
            
            # This is inherently sequential due to dependencies, but we can try to vectorize
            # Alternative: use the divisor-sum sieve approach with NumPy
            # sigma[k] = sum of all divisors of k
            # This can be done as: for d in 1..K_MAX: for j in d, 2d, 3d, ...: sigma[j] += d
            # But this is O(N log N) and might be slower for large N
            
            # Let's try the sequential approach but with some optimizations
            # Actually, for K_MAX up to 100M, the sequential loop in Python is too slow
            # Let's use the divisor sieve approach with NumPy
            
            # Reset sigma
            sigma = np.zeros(K_MAX + 1, dtype=np.int64)
            
            # Divisor sieve: for each d, add d to all multiples
            # This is O(N log N) but vectorizable
            t_sieve_start = time.time()
            for d in range(1, K_MAX + 1):
                if time.time() > deadline - 20:
                    break
                # Add d to sigma[d], sigma[2d], sigma[3d], ...
                multiples = np.arange(d, K_MAX + 1, d)
                sigma[multiples] += d
            
            t_sieve_end = time.time()
            
            if time.time() > deadline - 15:
                continue
            
            # Now find collisions: group by M = k * sigma[k]
            # Only consider k where M <= 10^18 and k <= isqrt(M)
            
            # Compute M values
            ks = np.arange(1, K_MAX + 1, dtype=np.int64)
            M_vals = ks * sigma[1:]  # M_vals[i] = (i+1) * sigma[i+1]
            
            # Filter: M <= 10^18 and k <= isqrt(M)
            # k <= isqrt(M) means k^2 <= M, i.e., k^2 <= k*sigma(k), i.e., k <= sigma(k)
            # This is always true for k >= 1 since sigma(k) >= k+1 for k > 1, and sigma(1)=1
            # Wait: sigma(1) = 1, so k=1: M=1, isqrt(1)=1, k=1 <= 1. OK.
            # For k > 1, sigma(k) >= k + 1 > k, so k < sigma(k), so k^2 < k*sigma(k) = M, so k < sqrt(M).
            # So the constraint k <= isqrt(M) is automatically satisfied for all k >= 1.
            # Let me verify: k=2, sigma(2)=3, M=6, isqrt(6)=2, k=2 <= 2. OK.
            # k=3, sigma(3)=4, M=12, isqrt(12)=3, k=3 <= 3. OK.
            # k=4, sigma(4)=7, M=28, isqrt(28)=5, k=4 <= 5. OK.
            # So yes, k <= isqrt(k*sigma(k)) is always true.
            
            valid_mask = M_vals <= 10**18
            if not np.any(valid_mask):
                continue
                
            valid_ks = ks[valid_mask]
            valid_Ms = M_vals[valid_mask]
            
            # Group by M value
            # Sort by M, then find consecutive groups
            order = np.argsort(valid_Ms, kind='mergesort')
            sorted_Ms = valid_Ms[order]
            sorted_ks = valid_ks[order]
            
            # Find group boundaries
            # Where sorted_Ms changes, a new group starts
            if len(sorted_Ms) == 0:
                continue
                
            # Use diff to find where M changes
            changes = np.diff(sorted_Ms) != 0
            # Group starts: index 0 and where changes is True
            group_starts = np.concatenate([[0], np.where(changes)[0] + 1])
            group_ends = np.concatenate([group_starts[1:], [len(sorted_Ms)]])
            
            # Find the group with the most elements
            group_sizes = group_ends - group_starts
            max_idx = np.argmax(group_sizes)
            max_size = group_sizes[max_idx]
            
            if max_size > best_count:
                start = group_starts[max_idx]
                end = group_ends[max_idx]
                candidate_ks = sorted_ks[start:end]
                candidate_M = sorted_Ms[start]
                
                # Verify with exact sigma computation
                verified_ks = []
                for k in candidate_ks:
                    k_int = int(k)
                    if k_int * _sigma_exact(k_int) == int(candidate_M) and k_int <= math.isqrt(int(candidate_M)):
                        verified_ks.append(k_int)
                
                if len(verified_ks) > best_count:
                    best_M = int(candidate_M)
                    best_ks = verified_ks
                    best_count = len(verified_ks)
                    
        except Exception:
            break
            
    # Final type conversion and verification
    final_ks = []
    for k in best_ks:
        k_int = int(k)
        if isinstance(k_int, int) and not isinstance(k_int, bool):
            if 1 <= k_int <= math.isqrt(best_M) and k_int * _sigma_exact(k_int) == best_M:
                if k_int not in final_ks:
                    final_ks.append(k_int)
    
    if len(final_ks) == 0:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': int(best_M), 'k': final_ks}

def _sigma_exact(n):
    """Exact sigma(n) via trial division."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    s = 1
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            term = 1
            pk = 1
            while x % p == 0:
                x //= p
                pk *= p
                term += pk
            s *= term
        p += 1 if p == 2 else 2
    if x > 1:
        s *= (1 + x)
    return s
