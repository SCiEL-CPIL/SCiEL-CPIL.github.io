"""Find M with many k such that k*sigma(k)=M, k<=sqrt(M)."""
import time
import math
from math import isqrt
from collections import defaultdict

def _sigma(n):
    """Compute sigma(n) by trial division."""
    if n <= 1:
        return n
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            s = 1
            pk = 1
            while temp % d == 0:
                temp //= d
                pk *= d
                s += pk
            result *= s
        d += 1
    if temp > 1:
        result *= (1 + temp)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Strategy: enumerate k from 1 to K_max, compute M = k*sigma(k),
    # group by M, and find M with most k values where k <= sqrt(M).
    
    # For a given M, k must satisfy k <= sqrt(M) = sqrt(k*sigma(k)),
    # i.e., k^2 <= k*sigma(k), i.e., k <= sigma(k).
    # This is always true for k >= 1 (sigma(k) >= k+1 for k>1, sigma(1)=1).
    # Wait: sigma(1) = 1, so k=1: 1 <= 1, OK.
    # For k>1: sigma(k) >= k+1 > k, so k < sigma(k), thus k < sqrt(k*sigma(k)).
    # Actually: k <= sqrt(k*sigma(k)) iff k^2 <= k*sigma(k) iff k <= sigma(k).
    # For k=1: 1 <= 1, equality holds, so k=1 is exactly at the boundary.
    # For k>1: sigma(k) >= k+1, so k < sigma(k), strictly inside.
    
    # So ALL k >= 1 satisfy k <= sqrt(k*sigma(k))!
    # This means we just need to find M with many distinct k mapping to it.
    
    # Enumerate k and group by M = k * sigma(k)
    M_to_ks = defaultdict(list)
    
    # How far can we go? k*sigma(k) <= 10^18
    # sigma(k) is roughly k * (some factor), so k^2 * factor <= 10^18
    # For k up to ~10^9, sigma(k) ~ k, so k*sigma(k) ~ k^2 ~ 10^18
    # But we can't enumerate up to 10^9.
    
    # Let's enumerate k up to a reasonable limit and see what we find.
    # For small k, sigma(k) is small, so M = k*sigma(k) is small.
    # Collisions are more likely for small M.
    
    # Let's go up to k = 200000 first
    K_max = 200000
    
    for k in range(1, K_max + 1):
        if time.time() > deadline - 5:
            break
        s = _sigma(k)
        M = k * s
        if M > 10**18:
            # sigma is increasing on average, but not monotonic.
            # For very large k, M will exceed 10^18.
            # But we should still check a few more since sigma isn't monotonic.
            # Actually, for k > 10^9, k*sigma(k) > 10^18 almost certainly.
            # Let's just break if k is large enough.
            if k > 10**8:
                break
            continue
        M_to_ks[M].append(k)
    
    # Find the M with the most k values
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    for M, ks in M_to_ks.items():
        if len(ks) > best_count:
            # Verify all k satisfy k <= sqrt(M)
            valid_ks = [k for k in ks if k <= isqrt(M)]
            if len(valid_ks) > best_count:
                best_count = len(valid_ks)
                best_M = M
                best_ks = valid_ks
    
    # If we didn't find anything better, try a targeted search
    # for larger k values that might produce collisions
    
    if best_count <= 3 and time.time() < deadline - 10:
        # Try specific ranges of k where collisions might occur
        # Focus on k with similar prime factorizations
        
        # Try k in ranges where sigma(k) might collide
        # e.g., k and k' with same prime set but different exponents
        
        # Let's try k from 100000 to 500000 in steps
        for k in range(100000, 500001):
            if time.time() > deadline - 5:
                break
            s = _sigma(k)
            M = k * s
            if M > 10**18:
                break
            if M in M_to_ks:
                M_to_ks[M].append(k)
            else:
                M_to_ks[M] = [k]
        
        # Re-check
        for M, ks in M_to_ks.items():
            if len(ks) > best_count:
                valid_ks = [k for k in ks if k <= isqrt(M)]
                if len(valid_ks) > best_count:
                    best_count = len(valid_ks)
                    best_M = M
                    best_ks = valid_ks
    
    # Final verification
    for k in best_ks:
        assert isinstance(k, int)
        assert k * _sigma(k) == best_M, f"Failed: k={k}, sigma={_sigma(k)}, M={best_M}"
        assert k <= isqrt(best_M)
        assert 1 <= best_M <= 10**18
    
    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
