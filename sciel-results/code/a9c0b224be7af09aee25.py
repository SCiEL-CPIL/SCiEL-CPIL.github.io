"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def _sigma(n):
    """Compute sigma(n) = sum of divisors of n."""
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            term = 1
            pk = 1
            while temp % p == 0:
                temp //= p
                pk *= p
                term += pk
            result *= term
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (1 + temp)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Strategy: search ranges of k, compute k*sigma(k), group by M
    # Try multiple ranges to find dense clusters
    
    # First, try smaller ranges with exact computation
    ranges_to_try = [
        (100, 500),
        (500, 2000),
        (2000, 10000),
        (10000, 50000),
        (50000, 200000),
        (200000, 1000000),
        (1000000, 5000000),
    ]
    
    for lo, hi in ranges_to_try:
        if time.time() > deadline - 5:
            break
        
        # Compute k*sigma(k) for k in [lo, hi]
        groups = defaultdict(list)
        
        for k in range(lo, hi + 1):
            s = _sigma(k)
            M = k * s
            if M <= 10**18 and k <= math.isqrt(M):
                groups[M].append(k)
        
        # Check for best M in this range
        for M, ks in groups.items():
            if len(ks) > best_count:
                best_M = M
                best_k = ks
                best_count = len(ks)
        
        # If we found a good cluster, try to expand around it
        if best_count >= 5 and time.time() > deadline - 30:
            break
    
    # Try to find even more solutions by checking around known good M values
    # Use a wider search if time permits
    if time.time() < deadline - 20:
        # Try to find M with many solutions by searching more systematically
        # Focus on ranges where k*sigma(k) might cluster
        # sigma(k) ~ k * pi^2/6 * (1/k) for random k, so k*sigma(k) ~ k^2 * constant
        # This means k*sigma(k) is roughly proportional to k^2
        
        # Try to find M values that have many k preimages
        # by checking if we can find k1, k2, ... where k_i * sigma(k_i) = M
        
        # A different approach: for a given M, find all k such that k*sigma(k) = M
        # This means sigma(k) = M/k, so k must divide M and sigma(k) = M/k
        
        # Let's try to construct M with many solutions
        # If we pick k values that are all divisors of some M and have the right sigma
        pass
    
    # Final validation and return
    # Verify all solutions
    valid_ks = []
    for k in best_k:
        if isinstance(k, int) and not isinstance(k, bool):
            if 1 <= k <= math.isqrt(best_M):
                s = _sigma(k)
                if k * s == best_M:
                    valid_ks.append(k)
    
    # Ensure distinct
    valid_ks = list(set(valid_ks))
    valid_ks.sort()
    
    if len(valid_ks) == 0:
        # Fallback to known solution
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': int(best_M), 'k': [int(k) for k in valid_ks]}
# EVOLVE-BLOCK-END
