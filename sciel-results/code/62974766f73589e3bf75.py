"""Optimized program using single-pass sieve for largest prime factor."""

# EVOLVE-BLOCK-START
def construct():
    N = 123457
    # lp[i] will store the largest prime factor of i
    # Initialize with 0 to indicate "not set" / prime candidate
    lp = [0] * N
    lp[1] = 1
    
    # Single pass sieve for Largest Prime Factor
    # If lp[i] is 0, i is prime. We set lp[i]=i and update all multiples.
    # Since we iterate i in increasing order, the last prime to update a multiple
    # is the largest prime factor of that multiple.
    for i in range(2, N):
        if lp[i] == 0:
            lp[i] = i
            for j in range(i, N, i):
                lp[j] = i
    
    # Find the minimum of max(lp[a], lp[N-a]) for a in [1, N-1]
    best_val = float('inf')
    best_a = 1
    
    for a in range(1, N):
        b = N - a
        val = lp[a] if lp[a] > lp[b] else lp[b]
        if val < best_val:
            best_val = val
            best_a = a
    
    a = best_a
    b = N - a
    return [a, b]
# EVOLVE-BLOCK-END
