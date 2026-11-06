"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    
    N = 123457
    
    # Sieve to compute largest prime factor for all numbers up to N
    # lp[n] = largest prime factor of n
    lp = [0] * (N + 1)
    lp[0] = 0
    lp[1] = 1
    
    # Sieve of Eratosthenes variant: for each prime p, mark all multiples
    # and set lp[multiple] = p (overwriting with larger primes as we go)
    # Actually, we want the LARGEST prime factor, so we process primes in order
    # and for each prime p, set lp[k*p] = p for all k. Since we go from small to large,
    # the last assignment for each number will be its largest prime factor.
    
    # More efficient: initialize lp[i] = i for all i (assuming prime), then
    # for each prime p, for each multiple m of p, set lp[m] = max(lp[m], p)
    # But simpler: just iterate primes and assign lp[m] = p for multiples m.
    # Since primes are processed in increasing order, the final lp[m] is the largest prime factor.
    
    # Standard approach:
    # lp[i] = i for all i >= 2
    # For p from 2 to sqrt(N):
    #   if lp[p] == p (p is prime):
    #     for j from p*p to N, step p:
    #       if lp[j] == j:  # j hasn't been marked yet? No, we want largest.
    #       lp[j] = p  # This would give smallest, not largest.
    
    # To get LARGEST prime factor:
    # We can do: for p from 2 to N:
    #   if lp[p] == 0 (or p is prime):
    #     for j from p to N, step p:
    #       lp[j] = p  # overwrites, so last (largest) prime factor wins
    
    # But this is O(N log log N) which is fine for N=123457.
    
    # Let's use: initialize lp[i] = 1 for all i
    # For each prime p (found via sieve), for each multiple m of p, set lp[m] = p
    # Since we process primes in increasing order, lp[m] ends up as the largest prime factor.
    
    # Actually, even simpler: just mark it directly.
    lp = [1] * (N + 1)
    # lp[0] = 0, lp[1] = 1 (already set)
    
    # Sieve for primes and assign largest prime factor
    # We need to identify primes first, or just iterate all numbers.
    # For i from 2 to N:
    #   if lp[i] == 1:  # i is prime (no smaller prime factor marked it... wait)
    
    # Let me think again. The cleanest way:
    # Create an array lp of size N+1, initialized to 1.
    # For p from 2 to N:
    #   if lp[p] == 1:  # p is prime
    #     for j in range(p, N+1, p):
    #       lp[j] = p
    
    # This works because when we reach prime p, we set lp[j] = p for all multiples j.
    # Since primes are processed in increasing order, the last prime that divides j
    # (i.e., the largest) will be the one that last wrote to lp[j].
    
    for p in range(2, N + 1):
        if lp[p] == 1:  # p is prime
            for j in range(p, N + 1, p):
                lp[j] = p
    
    # Now lp[n] is the largest prime factor of n for all n in [2, N]
    # lp[1] = 1
    
    # Now scan a from 1 to N-1, b = N - a
    # We want to minimize max(lp[a], lp[b])
    # Since a+b=N, both a and b are in [1, N-1], so lp is defined.
    
    best_val = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        b = N - a
        val = max(lp[a], lp[b])
        if val < best_val:
            best_val = val
            best_a = a
            best_b = b
    
    # Check time
    if time.time() > deadline:
        # Shouldn't happen, but safety
        pass
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
