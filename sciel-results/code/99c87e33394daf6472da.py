"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    """
    Find the longest increasing sequence of primes <= 10000 with nondecreasing gaps.
    
    Strategy:
    1. Generate all primes <= 10000.
    2. Use dynamic programming / search to find the longest subsequence
       where successive gaps are nondecreasing.
       
    Key insight: For a fixed starting pair (p1, p2), the greedy choice of
    the smallest valid next prime is optimal for maximizing length, because
    a smaller next prime leaves more room for subsequent primes.
    
    We try all starting pairs and use memoized greedy extension.
    """
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    # Step 1: Generate all primes <= 10000 using sieve
    limit = 10000
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i in range(2, limit + 1) if sieve[i]]
    n = len(primes)
    
    # Create a set for O(1) prime lookup
    prime_set = set(primes)
    
    # Step 2: For each starting pair, greedily extend
    # The greedy rule: given current last two primes (a, b), the next prime c
    # must satisfy c - b >= b - a. To maximize length, pick the SMALLEST such c.
    
    # Precompute: for each prime, find next prime (for quick lookup)
    # Actually, we need to find the smallest prime >= (b + (b-a))
    
    # Build a list of primes and use binary search
    import bisect
    
    best_seq = []
    best_len = 0
    
    # Try all starting pairs
    # Optimization: we can prune - if remaining primes can't beat best_len, skip
    for i in range(n - 1):
        # Check time
        if time.time() > deadline:
            break
        
        p1 = primes[i]
        # Upper bound on possible length from this start:
        # At most n - i - 1 more primes, so max length is n - i
        if n - i <= best_len:
            break
        
        for j in range(i + 1, n):
            # Check time periodically
            if time.time() > deadline:
                break
            
            p2 = primes[j]
            gap = p2 - p1
            
            # Upper bound: at most n - j - 1 more primes
            if n - j - 1 + 2 <= best_len:
                break
            
            # Greedy extension: start with [p1, p2], keep picking smallest valid next prime
            seq = [p1, p2]
            last_gap = gap
            last_prime = p2
            
            # Find smallest prime >= last_prime + last_gap
            while True:
                min_next = last_prime + last_gap
                # Binary search for smallest prime >= min_next
                idx = bisect.bisect_left(primes, min_next)
                if idx >= n:
                    break
                next_prime = primes[idx]
                new_gap = next_prime - last_prime
                # new_gap >= last_gap is guaranteed since next_prime >= last_prime + last_gap
                seq.append(next_prime)
                last_gap = new_gap
                last_prime = next_prime
                
                # Check time
                if time.time() > deadline:
                    break
            
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
            
            # If we hit the deadline, break
            if time.time() > deadline:
                break
    
    # If no sequence found (shouldn't happen), return empty or minimal
    if not best_seq:
        # Fallback: just return [2, 3] or similar
        return [p for p in primes if p <= 10000][:2] if n >= 2 else primes
    
    return best_seq
# EVOLVE-BLOCK-END
