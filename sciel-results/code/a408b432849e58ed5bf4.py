"""Program to find M with many solutions to k*sigma(k) = M."""
import math
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
def _sigma_sieve(limit):
    """Compute sigma(k) for all k up to limit using a sieve method.
    sigma(k) = sum of divisors of k.
    For each d, add d to sigma(multiple) for all multiples of d.
    """
    sigma = [0] * (limit + 1)
    for d in range(1, limit + 1):
        for m in range(d, limit + 1, d):
            sigma[m] += d
    return sigma

def _sigma_by_factorization(n):
    """Compute sigma(n) by trial factorization."""
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            power_sum = 1
            power_val = 1
            while temp % p == 0:
                temp //= p
                power_val *= p
                power_sum += power_val
            result *= power_sum
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (1 + temp)
    return result

def _verify_solution(k, M):
    """Verify that k * sigma(k) == M exactly."""
    s = _sigma_by_factorization(k)
    return k * s == M

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill
    
    # Phase 1: Sieve-based search over a range of k values
    # We compute k*sigma(k) for k up to some limit, then group by M value
    # and find the M with the most k's.
    
    # Start with a moderate limit and expand if time permits
    # k must be <= sqrt(M), and M = k*sigma(k)
    # For k up to K, M = k*sigma(k) can be up to about K * K * log(log(K))
    # We need M <= 10^18, so K * sigma(K) <= 10^18
    # sigma(K) ~ K*log(log(K)) for large K, so K^2 * log(log(K)) <= 10^18
    # K ~ 10^9 / sqrt(log(log(K))) ~ roughly 10^8 to 10^9
    
    # But we can only sieve up to a reasonable limit in the time available.
    # Let's try to sieve up to a large limit.
    
    # Strategy: try increasing limits until we run out of time
    # For each limit, compute all k*sigma(k) for k in [1, limit]
    # Group by M, find best M
    
    best_M = 333312
    best_k = [336, 372, 434]
    
    # Try a sequence of increasing limits
    limits = [10000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000]
    
    # Also try to go much higher with a more targeted approach
    # For very large k, we can't sieve, but we can check specific k values
    
    for limit in limits:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:  # Need 20s safety margin
            break
        
        # Compute sigma via sieve
        sigma = _sigma_sieve(limit)
        
        # Group k by M = k * sigma(k)
        groups = defaultdict(list)
        for k in range(1, limit + 1):
            M_val = k * sigma[k]
            if M_val <= 10**18:
                groups[M_val].append(k)
        
        # Find the M with the most k's (all k <= sqrt(M) automatically since k*sigma(k)=M implies k <= M, and we need k <= sqrt(M))
        # Actually we need k <= sqrt(M). Since M = k*sigma(k), we need k <= sqrt(k*sigma(k)), i.e., k <= sigma(k).
        # sigma(k) >= k+1 for k >= 1 (sigma(k) >= 1 + k), so k < sigma(k) always, meaning k < sqrt(k*sigma(k)) = sqrt(M).
        # Wait: k <= sqrt(M) means k^2 <= M = k*sigma(k), so k <= sigma(k). Since sigma(k) >= k+1 > k for k>=1, this always holds.
        # So all k in our range automatically satisfy k <= sqrt(M).
        
        # Find best group
        for M_val, k_list in groups.items():
            if len(k_list) > len(best_k):
                best_M = M_val
                best_k = k_list
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
    
    # Phase 2: If we have time, try to search for larger k values individually
    # For large k, sigma(k) is roughly k * log(log(k)), so M = k*sigma(k) ~ k^2 * log(log(k))
    # We want to find M values that have many preimages.
    # This is harder to do efficiently without a full sieve.
    # Instead, let's try to extend our sieve limit if we have time.
    
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30:
        # Try a much larger sieve
        large_limit = 10000000  # 10 million
        if elapsed + 10 < DEADLINE:  # rough estimate
            try:
                sigma = _sigma_sieve(large_limit)
                groups = defaultdict(list)
                for k in range(1, large_limit + 1):
                    M_val = k * sigma[k]
                    if M_val <= 10**18:
                        groups[M_val].append(k)
                
                for M_val, k_list in groups.items():
                    if len(k_list) > len(best_k):
                        best_M = M_val
                        best_k = k_list
            except MemoryError:
                pass
    
    # Verify all solutions
    verified_k = []
    for k in best_k:
        if _verify_solution(k, best_M):
            verified_k.append(k)
    
    # Ensure k values are distinct and sorted
    verified_k = list(set(verified_k))
    verified_k.sort()
    
    # If verification failed for some, fall back to known good answer
    if len(verified_k) < 3:
        best_M = 333312
        verified_k = [336, 372, 434]
    
    # Ensure types are correct
    best_M = int(best_M)
    verified_k = [int(k) for k in verified_k]
    
    return {'M': best_M, 'k': verified_k}
# EVOLVE-BLOCK-END
