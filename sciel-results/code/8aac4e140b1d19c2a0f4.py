# EVOLVE-BLOCK-START
import time
import array

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 123457
    max_n = N - 1
    
    # Sieve for largest prime factor using array of unsigned ints for efficiency
    lpf = array.array('I', [0]) * (max_n + 1)
    lpf[1] = 1
    
    for p in range(2, max_n + 1):
        if time.time() > deadline:
            # Fallback: return trivial decomposition if sieve takes too long
            return [1, N - 1]
        if lpf[p] == 0:  # p is prime
            for k in range(p, max_n + 1, p):
                lpf[k] = p
    
    # Find a that minimizes max(P(a), P(N-a))
    # Search only a in [1, N//2] due to symmetry
    best_max = N + 1
    best_a = 1
    best_b = N - 1
    
    half = N // 2
    for a in range(1, half + 1):
        if (a & 0xFF) == 0 and time.time() > deadline:
            break
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        m = pa if pa > pb else pb
        if m < best_max:
            best_max = m
            best_a = a
            best_b = b
            # Early termination if we reach theoretical minimum
            # Since N is odd, one of a,b is even (P>=2), other is odd (P>=3 if >1)
            if best_max <= 3:
                break
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
