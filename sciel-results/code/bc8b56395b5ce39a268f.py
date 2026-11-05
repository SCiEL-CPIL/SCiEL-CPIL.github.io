"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0  # 85s out of 90s budget
    N = 123457
    
    # Sieve to compute largest prime factor for all numbers up to N
    # This is O(N log log N) and very fast
    lpf = list(range(N + 1))  # lpf[i] will hold largest prime factor of i
    
    # Sieve of Eratosthenes variant: for each prime p, update lpf for multiples
    for i in range(2, N + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i  # i is the largest prime factor seen so far for j
                # Since we go in increasing order, the last prime that divides j is the largest
    
    # Now lpf[x] holds the largest prime factor of x for all x in [1, N]
    # lpf[1] = 1 (P(1) = 1)
    
    # Find a, b with a + b = N, minimizing max(lpf[a], lpf[b])
    best_max = N + 1
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        m = pa if pa > pb else pb
        if m < best_max:
            best_max = m
            best_a = a
            best_b = b
            # Check deadline
            if time.time() > deadline:
                break
            # If best_max is very small, we can stop early
            # The minimum possible value of max(P(a), P(b)) is at least 2
            # (since both a and b are >= 1, and if one is 1, P=1, but the other
            #  has P >= 2 unless it's also 1, which isn't possible here)
            # Actually, if a=1, b=N, P(b) could be large.
            # The absolute minimum is 2, achieved only if both a and b are
            # powers of 2 (or 1). Since a+b=N (odd), one is even, one is odd.
            # For both to have P=2, the odd one must be 1. So a=1,b=N or a=N,b=1.
            # P(N) is the largest prime factor of 123457.
            # 123457 = ? Let's not assume, the loop will find it.
            if best_max == 2:
                break
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
