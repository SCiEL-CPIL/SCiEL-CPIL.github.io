import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    LO = 10000
    HI = 1000000
    MAX_VAL = HI + 1
    
    # Sieve for Largest Prime Factor
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (MAX_VAL + 1)
    
    # We iterate p from 2 to MAX_VAL
    # If lpf[p] is 0, p is prime
    for p in range(2, MAX_VAL + 1):
        if lpf[p] == 0:
            # p is prime, update all multiples
            # Since we go in increasing order of p, the last prime that hits a number is its largest prime factor
            for m in range(p, MAX_VAL + 1, p):
                lpf[m] = p
                
    # Check deadline
    if time.time() > deadline:
        # Fallback to a reasonable guess if sieve took too long (unlikely)
        return 10000

    # Find n that minimizes max(lpf[n], lpf[n+1])
    best_n = LO
    best_score = float('inf')
    
    for n in range(LO, HI + 1):
        score = lpf[n] if lpf[n] > lpf[n+1] else lpf[n+1]
        if score < best_score:
            best_score = score
            best_n = n
            # If we find a score of 2, that's the theoretical minimum (only prime factor 2)
            # But n and n+1 cannot both be powers of 2 (except 1 and 2, which is out of range).
            # So we just keep searching.
            
    return best_n
