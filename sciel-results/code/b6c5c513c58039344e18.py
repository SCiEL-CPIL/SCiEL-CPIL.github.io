import time

def construct():
    """
    Returns two positive integers a, b such that a + b = 123457 
    and max(P(a), P(b)) is minimized, where P(n) is the largest prime factor of n.
    """
    target = 123457
    
    # Precompute largest prime factor for all numbers up to target
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (target + 1)
    
    # Sieve-like approach to find largest prime factors
    # Initialize lpf[i] = i for all i (assuming i is prime for now, or just a placeholder)
    # Actually, we can iterate i from 2 to target.
    # If lpf[i] is 0, i is prime.
    # Then for all multiples j of i, set lpf[j] = i.
    # Since we iterate i in increasing order, the last prime that divides j will be the largest.
    
    for i in range(2, target + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, target + 1, i):
                lpf[j] = i
                
    # P(1) = 1
    lpf[1] = 1
    
    best_score = float('inf')
    best_a = 1
    best_b = target - 1
    
    # Iterate through all possible splits
    # Since a+b is symmetric, we only need to check a up to target//2
    for a in range(1, target // 2 + 1):
        b = target - a
        pa = lpf[a]
        pb = lpf[b]
        score = pa if pa > pb else pb
        
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            
    return [best_a, best_b]
