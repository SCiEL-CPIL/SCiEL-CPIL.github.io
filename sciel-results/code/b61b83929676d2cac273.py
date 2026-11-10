import time
import math

def construct():
    N = 123457
    
    # Precompute Largest Prime Factor (LPF) for all numbers up to N
    # Using a sieve-like approach: iterate primes in increasing order,
    # marking multiples. Since we go in increasing order, the final value
    # of lpf[j] will be the largest prime factor of j.
    lpf = [0] * (N + 1)
    lpf[1] = 1  # P(1) = 1
    
    for p in range(2, N + 1):
        if lpf[p] == 0:
            # p is prime
            for j in range(p, N + 1, p):
                lpf[j] = p
    
    # Now lpf[i] holds the largest prime factor of i.
    
    best_score = float('inf')
    best_a = 1
    best_b = N - 1
    
    # Search for a in [1, N//2]
    # We only need to check up to N//2 because (a,b) and (b,a) are symmetric in max(P(a),P(b))
    mid = N // 2
    
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    for a in range(1, mid + 1):
        if time.time() > deadline:
            break
        b = N - a
        score = max(lpf[a], lpf[b])
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            
    return [best_a, best_b]
