import math
import time
from collections import defaultdict


def _sigma_sieve(n):
    """
    Compute sigma(1), sigma(2), ..., sigma(n) using a sieve.
    sigma(k) = sum of divisors of k.
    Uses the identity: for each divisor d, it contributes d to sigma(d), sigma(2d), sigma(3d), ...
    More efficiently: use the multiplicative property via prime factorization.
    
    Actually, the fastest sieve approach:
    Initialize sigma = [1] * (n+1)
    For each prime p, for each multiple m of p, multiply sigma[m] by (1 + p + p^2 + ... + p^a)
    where p^a || m.
    
    Simpler approach: 
    sigma[0..n] = [1] * (n+1)
    For d from 1 to n:
        For m from d to n step d:
            sigma[m] += d
    This is O(n log n) which for n=10^7 is ~10^7 * ln(10^7) ~ 1.6*10^8 operations.
    In Python this is too slow.
    
    Better: use the prime-based multiplicative sieve.
    """
    import numpy as np
    
    if n < 1:
        return np.zeros(0, dtype=np.int64)
    
    sigma = np.ones(n + 1, dtype=np.int64)
    # sigma[0] = 0
    sigma[0] = 0
    
    # Sieve of Eratosthenes to find primes up to n
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # For each prime p, update sigma for all multiples
    # For a number m = p^a * rest (p not dividing rest), sigma(m) = sigma(p^a) * sigma(rest)
    # sigma(p^a) = (p^(a+1) - 1) / (p - 1)
    
    # We'll use a different approach: 
    # sigma[m] starts at 1. For each prime p dividing m, we need to multiply by the appropriate factor.
    # But tracking exponents is complex. Let's use the additive sieve but vectorized.
    
    # Actually for n up to ~5*10^6, the additive sieve in numpy is feasible:
    # sigma[m] = sum of d for d|m
    # For d in 1..n: for m in d, 2d, 3d, ...: sigma[m] += d
    
    # This is O(n log n) total additions. For n=5*10^6, that's about 5*10^6 * 15 = 7.5*10^7 operations.
    # In numpy vectorized, each step d does n//d additions. Total = n * H(n) ~ n * ln(n).
    
    # Let's do it in chunks to manage memory
    for d in range(1, n + 1):
        if d % 1000000 == 0:
            pass  # could check time here
        sigma[d::d] += d
    
    return sigma


def _sigma_sieve_fast(n, deadline):
    """
    Faster sieve for sigma using prime factorization approach.
    Uses the multiplicative property.
    """
    import numpy as np
    
    if n < 1:
        return np.zeros(0, dtype=np.int64)
    
    # Use the classic sieve:
    # sigma[i] = 1 initially
    # For each i from 2 to n, if sigma[i] == 1 (i is prime), then for all multiples j of i,
    # we need to account for the prime power factor.
    
    # Alternative: use the formula that for each d, it contributes to sigma[d], sigma[2d], ...
    # But vectorized in numpy this is the fastest.
    
    sigma = np.ones(n + 1, dtype=np.int64)
    sigma[0] = 0
    
    # Check time periodically
    for d in range(1, n + 1):
        if d % 2000000 == 0:
            if time.time() > deadline:
                # Return partial result - but this breaks correctness
                # Better to not go this far
                pass
        sigma[d::d] += d
    
    return sigma


def construct():
    """
    Find an integer M <= 10^18 with many distinct k_i <= sqrt(M) such that k_i * sigma(k_i) = M.
    
    Strategy: Use a sieve to compute sigma(k) for all k up to N, then find M values
    with the most k's such that k*sigma(k) = M.
    """
    start_time = time.time()
    deadline = start_time + 115  # Leave margin before 150s kill
    
    import numpy as np
    
    # Start with the known seed result
    M_seed = 333312
    k_list_seed = []
    sqrt_M = math.isqrt(M_seed)
    
    # Quick sigma for seed verification (trial division for small numbers)
    def sigma_small(n):
        if n <= 1:
            return n
        result = 1
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                term = 1
                while temp % p == 0:
                    temp //= p
                    exp += 1
                    term += p ** exp
                result *= term
            p += 1
        if temp > 1:
            result *= (1 + temp)
        return result
    
    for k in range(1, sqrt_M + 1):
        if k * sigma_small(k) == M_seed:
            k_list_seed.append(k)
    
    best_M = M_seed
    best_k = list(k_list_seed)
    
    # Now do the main sieve-based search
    # We need to find N such that we can compute sigma up to N within time budget
    # The sieve sigma[d::d] += d for d=1..n takes about n*ln(n) numpy operations
    # For n=10^7: ~1.6*10^8 operations, each is a numpy vector op
    # Let's estimate: numpy can do ~10^8 simple ops/sec, so ~1.6 seconds for n=10^7
    # But the slice assignment sigma[d::d] += d creates a view and does element-wise add
    # Total work = sum_{d=1}^{n} n/d = n * H(n) ~ n * (ln(n) + 0.577)
    # For n = 5*10^6: 5*10^6 * (15.4 + 0.577) ~ 8*10^7 element updates
    # For n = 10^7: 10^7 * (16.1 + 0.577) ~ 1.67*10^8 element updates
    
    # Let's try n = 10^7 first, and if time permits, go higher
    # Actually, let's be more careful. Each sigma[d::d] += d involves:
    # - Creating a strided view (O(1))
    # - Adding d to each element in-place: n//d operations
    # Total operations: sum_{d=1}^{n} floor(n/d) ~ n * ln(n)
    
    # For n=10^7 in numpy int64, this should take maybe 2-5 seconds
    # Let's try up to 10^7
    
    target_n = 10_000_000
    
    # Check if we have enough time
    # Estimate: n * ln(n) / 10^8 seconds (rough)
    estimated_time = target_n * math.log(target_n) / 1e8
    if estimated_time > 80:
        # Reduce target
        target_n = int(80 * 1e8 / math.log(80 * 1e8))
        target_n = min(target_n, 10_000_000)
    
    # Compute sigma up to target_n
    sigma_arr = np.ones(target_n + 1, dtype=np.int64)
    sigma_arr[0] = 0
    
    # Sieve: for each d, add d to sigma[d], sigma[2d], sigma[3d], ...
    # We'll do this in a loop but check time
    last_check = time.time()
    for d in range(1, target_n + 1):
        if d % 1_000_000 == 0:
            if time.time() > deadline - 20:
                target_n = d - 1
                break
        sigma_arr[d::d] += d
    
    # Now sigma_arr[k] = sigma(k) for k = 0..target_n
    # Compute M_k = k * sigma(k) for k = 1..target_n
    # Filter: M_k <= 10^18 and k <= sqrt(M_k)
    # k <= sqrt(k*sigma(k)) means k^2 <= k*sigma(k), i.e., k <= sigma(k)
    # This is always true for k >= 1 since sigma(k) >= k+1 > k for k >= 2, and sigma(1)=1=1
    
    # Wait: k <= sqrt(M) where M = k*sigma(k)
    # k <= sqrt(k*sigma(k))
    # k^2 <= k*sigma(k)
    # k <= sigma(k)
    # For k=1: sigma(1)=1, 1<=1 ✓
    # For k>=2: sigma(k) >= 1+k > k ✓
    # So ALL k satisfy k <= sqrt(k*sigma(k))! Great, no filtering needed.
    
    # Now group by M_k = k * sigma(k)
    # k * sigma(k) can be up to 10^7 * 10^7 * ~log(10^7) ~ 10^15, well within 10^18
    
    k_range = np.arange(1, target_n + 1, dtype=np.int64)
    M_vals = k_range * sigma_arr[1:target_n+1]
    
    # Filter M_vals <= 10^18
    mask = M_vals <= 10**18
    M_filtered = M_vals[mask]
    k_filtered = k_range[mask]
    
    # Find the M with the most k values
    # Use a dict to count
    m_to_k = defaultdict(list)
    
    # For efficiency, sort by M and count
    # Or use numpy unique with return_inverse
    if len(M_filtered) > 0:
        # Find the mode (most frequent value)
        # For large arrays, use a hash-based approach
        # Actually, let's just iterate and use a dict
        # But for 10^7 elements, a Python dict might be slow
        
        # Alternative: sort and count
        # This is O(n log n)
        order = np.argsort(M_filtered, kind='stable')
        M_sorted = M_filtered[order]
        k_sorted = k_filtered[order]
        
        # Find runs of equal values
        if len(M_sorted) > 0:
            # Use diff to find boundaries
            diff = np.diff(M_sorted)
            boundaries = np.where(diff != 0)[0] + 1
            start_indices = np.concatenate([np.array([0]), boundaries])
            end_indices = np.concatenate([boundaries, np.array([len(M_sorted)])])
            
            # Count for each group
            counts = end_indices - start_indices
            
            # Find the group with max count
            max_idx = np.argmax(counts)
            max_count = int(counts[max_idx])
            best_M_candidate = int(M_sorted[start_indices[max_idx]])
            best_k_candidate = [int(k) for k in k_sorted[start_indices[max_idx]:end_indices[max_idx]]]
            
            # Also check the second and third most common for safety
            # But let's also look at all groups with count > current best
            current_best_count = len(best_k)
            
            if max_count > current_best_count:
                best_M = best_M_candidate
                best_k = best_k_candidate
            
            # Check other high-count groups
            # Get top few by count
            if len(counts) > 1:
                top_indices = np.argsort(counts)[::-1][:10]
                for idx in top_indices:
                    cnt = int(counts[idx])
                    if cnt > len(best_k):
                        m_val = int(M_sorted[start_indices[idx]])
                        k_vals = [int(k) for k in k_sorted[start_indices[idx]:end_indices[idx]]]
                        # Verify k <= sqrt(M)
                        sqrt_m = math.isqrt(m_val)
                        valid_ks = [k for k in k_vals if k <= sqrt_m]
                        if len(valid_ks) > len(best_k):
                            best_M = m_val
                            best_k = valid_ks
                    else:
                        break
    
    # Verify the result
    if best_k:
        # Double-check each k
        verified_k = []
        for k in best_k:
            if k * sigma_small(k) == best_M and k <= math.isqrt(best_M):
                verified_k.append(k)
        best_k = verified_k
    
    # Ensure types
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]
    best_k = sorted(set(best_k))
    best_k = best_k[:1000]
    
    return {'M': best_M, 'k': best_k}
