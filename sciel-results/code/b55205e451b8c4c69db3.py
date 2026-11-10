"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return two positive integers a, b with a + b = 123457 that minimize
    max(P(a), P(b)), where P(n) is the largest prime factor of n and P(1)=1.
    
    Strategy: Precompute largest prime factor for all numbers up to N=123457
    using a sieve, then scan all valid splits to find the global optimum.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    
    N = 123457
    
    # Sieve to compute largest prime factor for each number up to N
    # largest_prime_factor[i] will hold the largest prime factor of i
    largest_prime_factor = [1] * (N + 1)
    
    # Initialize: for primes, largest prime factor is themselves
    # Use a sieve approach: for each prime p, mark multiples of p
    # We want the LARGEST prime factor, so we process primes in increasing order
    # and overwrite, so the last (largest) prime factor wins
    
    # First, find all primes up to N using a simple sieve
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    primes = []
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    # Now, for each number, compute its largest prime factor
    # For each prime p, for each multiple m of p, set largest_prime_factor[m] = p
    # Since we process primes in increasing order, later (larger) primes overwrite
    # earlier ones, giving us the largest prime factor.
    # But this is O(N log log N * avg_multiples) which might be slow.
    
    # Better approach: for each number, find its largest prime factor by trial division
    # using the primes list. But that's O(N * pi(sqrt(N))) which is also slow.
    
    # Even better: use a modified sieve where for each prime p, we update multiples.
    # For each prime p, for m = p, 2p, 3p, ..., set lp[m] = p.
    # Since primes are processed in increasing order, the last write for each m
    # corresponds to the largest prime factor.
    
    lp = [1] * (N + 1)
    for p in primes:
        for m in range(p, N + 1, p):
            lp[m] = p
    
    # Now lp[n] is the largest prime factor of n for all n in [1, N]
    # lp[1] = 1 by initialization
    
    # Now scan all splits a + b = N where a, b >= 1
    # We want to minimize max(lp[a], lp[N-a])
    
    best_val = N  # upper bound: lp[N] <= N
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        b = N - a
        va = lp[a]
        vb = lp[b]
        m = va if va > vb else vb
        if m < best_val:
            best_val = m
            best_a = a
            best_b = b
            # If we've reached 2, that's the minimum possible (since P(1)=1, P(2)=2)
            # Actually the minimum max is at least 2 unless one is 1.
            # If one is 1, max is P(N). If both are composite with small factors...
            # The absolute minimum possible for max(P(a),P(b)) is 2 (if both are powers of 2, but they sum to odd, so one is even, one is odd; odd one must have odd prime factor >=3, so min is at least 3? Let's not assume, just keep scanning.)
            if best_val == 2:
                break
            if best_val == 3:
                # Can we do better? Only if both are powers of 2, but sum is odd, impossible.
                # Or one is 1 (P=1) and other is power of 3? No, P(power of 3) = 3.
                # So 3 might be optimal. But let's continue to be sure, or break if confident.
                # Actually, if a=1, b=N, max(1, P(N)). If N is not a power of 2 or 3, this is >3.
                # Let's just keep scanning; it's O(N) and fast.
                pass
    
    # Check deadline
    if time.time() > deadline:
        # Return best found so far
        pass
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
