# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 123457
    # Sieve for largest prime factor
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for p in range(2, N + 1):
        if time.time() > deadline:
            # Fallback if sieve takes too long (unlikely for N=123457)
            # But we need to return *something*. 
            # If we haven't finished sieve, we can't compute LPF accurately.
            # However, for this N, it's instant.
            pass 
        if lpf[p] == 0:  # p is prime
            for k in range(p, N + 1, p):
                lpf[k] = p
    
    # Find a that minimizes max(P(a), P(N-a))
    best_max = N + 1
    best_a = 1
    for a in range(1, N):
        if time.time() > deadline:
            break
        b = N - a
        m = lpf[a]
        if lpf[b] > m:
            m = lpf[b]
        if m < best_max:
            best_max = m
            best_a = a
    
    return [best_a, N - best_a]
# EVOLVE-BLOCK-END
