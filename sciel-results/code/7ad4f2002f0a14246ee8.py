"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit
    
    N = 1000000
    # We need lpf for indices up to N+1
    size = N + 2  # indices 0..N+1
    
    # Sieve to find largest prime factor for each number
    lpf = [0] * size
    # Initialize: for each number, its largest prime factor is at least itself if prime
    # We'll do this by iterating primes and updating multiples
    
    # First, mark primes and compute largest prime factor
    # Use a sieve approach: for each prime p, set lpf[k*p] = p for all k
    # This way, after processing all primes, lpf[n] holds the largest prime factor
    
    # We need primes up to N+1
    # Use standard sieve to get primes, then update lpf
    
    # Actually, more efficient: iterate from 2 to sqrt(N+1) for trial, 
    # but for largest prime factor, we can do:
    # lpf[n] = n initially (assuming n is prime)
    # For each i from 2 to sqrt(N+1):
    #   if lpf[i] == i (i is prime):
    #     for j from i*i to N+1 step i:
    #       lpf[j] = i  # but this gives smallest, not largest!
    
    # For LARGEST prime factor, better approach:
    # lpf[n] = n initially
    # For each i from 2 to N+1:
    #   if lpf[i] == i (i is prime):
    #     for j from i to N+1 step i:
    #       lpf[j] = i  # this will end up with the largest prime factor since we go in order
    # Wait, that's not right either. If we go in increasing order, the last prime that divides j will be the largest.
    
    # Better: for each prime p (in increasing order), for each multiple m of p, set lpf[m] = p.
    # Since primes are processed in increasing order, the last assignment is by the largest prime factor.
    
    # Let me use a different approach:
    # lpf = list(range(size))  # lpf[i] = i initially
    # for i in range(2, int(size**0.5) + 1):
    #   if lpf[i] == i:  # i is prime
    #     for j in range(i*i, size, i):
    #       lpf[j] = i  # This gives SMALLEST prime factor, not largest
    # 
    # For largest prime factor, I need a different strategy.
    # 
    # Alternative: For each number n, lpf[n] can be computed by:
    # - If n is prime, lpf[n] = n
    # - Otherwise, lpf[n] = max(lpf[n/p], p) where p is a prime factor
    #
    # Simpler: just use trial division for each number, but that's O(N * sqrt(N)) which is too slow.
    #
    # Best approach for largest prime factor via sieve:
    # lpf = [0] * size
    # for i in range(2, size):
    #   if lpf[i] == 0:  # i is prime
    #     for j in range(i, size, i):
    #       lpf[j] = i  # Since i increases, the last prime that divides j is the largest
    #
    # This is O(N log log N) approximately, which is fast enough for N=1000000.
    
    lpf = [0] * size
    for i in range(2, size):
        if lpf[i] == 0:  # i is prime
            for j in range(i, size, i):
                lpf[j] = i
                # Check deadline
                if (j % 1000000 == 0) and (time.time() > deadline):
                    break
            if time.time() > deadline:
                break
    
    if time.time() > deadline:
        # Fallback: return a reasonable default
        return 10000
    
    # Now find n in [10000, 1000000] that minimizes max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    for n in range(10000, N + 1):
        if time.time() > deadline:
            break
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
