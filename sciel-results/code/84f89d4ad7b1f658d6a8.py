import math
import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    # Known seed solution
    fallback = {'M': 333312, 'k': [336, 372, 434]}

    try:
        # Strategy: Enumerate k up to some N, compute M = k*sigma(k), bucket by M.
        # The key insight is that k*sigma(k) grows roughly as k^2 * pi^2/6, so
        # for a fixed M, k is bounded by roughly sqrt(6M/pi^2).
        # We want to find M with many k values mapping to it.
        #
        # To maximize collisions, we should look at k values where k*sigma(k) 
        # takes the same value. This happens when different k have the same
        # product k*sigma(k).
        #
        # For efficient computation, use a sieve for sigma up to N.
        # N can be up to ~10^7 in reasonable time with numpy.
        
        # Let's try N = 10^7
        N = 10 * 10**6
        
        # Check time budget
        if time.time() > deadline - 60.0:
            return fallback
        
        # Sieve for sigma
        sigma = np.ones(N + 1, dtype=np.int64)
        for i in range(1, N + 1):
            if time.time() > deadline - 30.0:
                # Time running out, use what we have
                N = i - 1
                break
            sigma[i::i] += i
        
        if N < 1000:
            return fallback
        
        # Compute M = k * sigma(k) for k = 1..N
        ks = np.arange(1, N + 1, dtype=np.int64)
        Ms = ks * sigma[1:N+1]
        
        # Filter: M must be <= 10^18 and k <= sqrt(M)
        # k <= sqrt(M) means k^2 <= M, i.e., k^2 <= k*sigma(k), i.e., k <= sigma(k)
        # This is always true for k >= 1 since sigma(k) >= k+1 > k for k >= 2, and sigma(1)=1, 1<=1.
        # Actually sigma(1) = 1, so k=1: 1 <= 1, OK.
        # For k >= 2, sigma(k) >= 1 + k > k, so k < sigma(k), hence k^2 < k*sigma(k) = M, so k < sqrt(M).
        # So the constraint k <= sqrt(M) is automatically satisfied for all k >= 1.
        # Wait, let me double check: k <= sqrt(M) iff k^2 <= M iff k^2 <= k*sigma(k) iff k <= sigma(k).
        # sigma(k) = sum of divisors of k. For k=1, sigma(1)=1, so 1<=1 OK.
        # For k>=2, sigma(k) >= 1 + k > k, so k < sigma(k), hence k^2 < M, so k < sqrt(M) < sqrt(M)+1.
        # So k <= sqrt(M) is satisfied.
        
        # Filter M <= 10^18
        valid_mask = Ms <= 10**18
        if not np.all(valid_mask):
            ks = ks[valid_mask]
            Ms = Ms[valid_mask]
        
        if len(ks) == 0:
            return fallback
        
        # Bucket by M
        # Use np.unique with return_inverse for efficient bincount
        unique_Ms, inverse = np.unique(Ms, return_inverse=True)
        counts = np.bincount(inverse)
        
        max_count = int(np.max(counts))
        
        if max_count <= 3:
            # No improvement over seed
            return fallback
        
        # Find all M with max_count (or close to it)
        # Pick the one with max_count
        best_indices = np.where(counts == max_count)[0]
        
        # Pick the first one (or we could pick one with smallest M, but any is fine)
        best_idx = int(best_indices[0])
        best_M = int(unique_Ms[best_idx])
        
        # Extract k values
        best_k_mask = (inverse == best_idx)
        best_ks = ks[best_k_mask]
        
        # Verify: all k <= sqrt(M) and k*sigma(k) = M
        # k <= sqrt(M) is guaranteed as shown above
        # k*sigma(k) = M is guaranteed by construction
        # But let's double check with exact integer arithmetic for safety
        # Convert to Python ints
        best_k_list = [int(k) for k in best_ks]
        
        # Verify each k
        verified_ks = []
        for k in best_k_list:
            # Compute sigma(k) exactly
            s = 0
            # Efficient sigma computation for single k
            limit = int(math.isqrt(k))
            for d in range(1, limit + 1):
                if k % d == 0:
                    s += d
                    if d != k // d:
                        s += k // d
            if k * s == best_M:
                if k <= math.isqrt(best_M) or k * k <= best_M:
                    verified_ks.append(k)
        
        if len(verified_ks) > len(fallback['k']):
            return {'M': best_M, 'k': verified_ks}
        else:
            return fallback
            
    except Exception as e:
        return fallback
