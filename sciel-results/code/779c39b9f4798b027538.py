"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find two positive integers a, b with a + b = 123457 that minimize max(P(a), P(b)),
    where P(n) is the largest prime factor of n (and P(1) = 1).
    
    Strategy:
    1. Precompute the largest prime factor for every number up to 123457 using a sieve.
    2. Scan all valid splits (a, 123457-a) and find the one minimizing max(P(a), P(b)).
    3. Return the optimal pair.
    
    This is exact and runs in O(N log log N) time for the sieve + O(N) for the scan.
    """
    target = 123457
    
    # Precompute largest prime factor for all numbers up to target
    # largest_prime_factor[i] = largest prime factor of i, or 1 if i is 1
    lpf = [0] * (target + 1)
    lpf[0] = 0
    lpf[1] = 1
    
    # Sieve: for each prime p, mark multiples with p as their largest prime factor
    # We iterate primes in increasing order, so the last prime that divides a number
    # is its largest prime factor.
    # More efficient: for each i from 2 to target, if lpf[i] == 0, it's a prime.
    # Then for each multiple j of i, set lpf[j] = i (since i is the largest prime
    # factor we've seen so far for j, and we go in increasing order... wait, that's
    # not right. We need the LARGEST prime factor.
    
    # Better approach: initialize lpf[i] = i for all i (assuming each is its own LPF).
    # Then for each prime p, for each multiple j of p, if p > lpf[j], update lpf[j] = p.
    # But that's O(N log N) which is fine for N=123457.
    
    # Actually, simpler: for each i from 2 to target:
    #   if lpf[i] == 0: i is prime
    #   for j in range(i, target+1, i):
    #       lpf[j] = i  # since we go in increasing order, the last assignment wins
    # This gives the largest prime factor because we overwrite with larger primes.
    
    # Wait, that's still O(N log N). Let me think again.
    # For each prime p, we visit N/p multiples. Sum over primes p <= N of N/p = N * sum(1/p) ~ N log log N.
    # That's fine.
    
    lpf = [0] * (target + 1)
    lpf[1] = 1
    
    for i in range(2, target + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, target + 1, i):
                lpf[j] = i  # i is a prime factor; since i increases, last write is largest
    
    # Now lpf[n] holds the largest prime factor of n for all n >= 2, and lpf[1] = 1.
    
    # Scan all splits
    best_score = float('inf')
    best_a = 1
    best_b = target - 1
    
    for a in range(1, target):
        b = target - a
        pa = lpf[a] if a > 1 else 1
        pb = lpf[b] if b > 1 else 1
        score = max(pa, pb)
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            # Early termination: if best_score == 2, that's the minimum possible
            # (since both a and b must be even, but their sum is odd, so one is odd...
            # wait, if score is 2, both a and b have largest prime factor <= 2,
            # meaning both are powers of 2. But sum of two powers of 2 is odd only
            # if one is 1 (2^0) and the other is odd... no, 2^k + 2^m is even for k,m >= 1.
            # So score 2 is impossible for an odd target. Minimum possible is 3.
            if best_score <= 2:
                break
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
