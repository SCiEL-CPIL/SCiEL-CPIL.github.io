import time

def construct():
    deadline = time.time() + 85.0
    TARGET = 123457
    
    # Sieve to compute largest prime factor for all integers up to TARGET
    lp = [0] * (TARGET + 1)
    lp[1] = 1  # P(1) = 1
    
    for i in range(2, TARGET + 1):
        if lp[i] == 0:  # i is prime
            for j in range(i, TARGET + 1, i):
                lp[j] = i  # i is the largest prime factor of j (since we go in increasing order)
    
    # Linear scan over all valid decompositions a + b = TARGET
    best_val = float('inf')
    best_a = 1
    best_b = TARGET - 1
    
    for a in range(1, TARGET):
        if time.time() > deadline - 2:
            break
        b = TARGET - a
        val = max(lp[a], lp[b])
        if val < best_val:
            best_val = val
            best_a = a
            best_b = b
            if best_val == 2:
                # Can't do better than 2 (smallest prime)
                break
    
    return [best_a, best_b]
