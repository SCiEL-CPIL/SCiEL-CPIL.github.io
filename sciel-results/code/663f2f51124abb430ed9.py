"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85
    N = 1000000
    T = 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    masks = [0] * (N + T + 1)
    for j, p in enumerate(primes):
        bit = 1 << j
        for m in range(p, N + T + 1, p):
            masks[m] |= bit
    
    best_score = num_primes + 1
    best_n = 1000
    
    counts = [0] * num_primes
    for i in range(1001, 1025):
        m = masks[i]
        for j in range(num_primes):
            if m & (1 << j):
                counts[j] += 1
    
    cur_score = sum(1 for c in counts if c > 0)
    if cur_score < best_score:
        best_score = cur_score
        best_n = 1000
    
    for n in range(1001, N + 1):
        if time.time() > deadline:
            break
        m_out = masks[n + 1]
        m_in = masks[n + 25]
        for j in range(num_primes):
            if m_out & (1 << j):
                counts[j] -= 1
            if m_in & (1 << j):
                counts[j] += 1
        
        cur_score = sum(1 for c in counts if c > 0)
        if cur_score < best_score:
            best_score = cur_score
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
