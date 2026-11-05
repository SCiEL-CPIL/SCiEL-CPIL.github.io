import math

def construct():
    """
    Returns an integer n <= 10^18 that maximizes the number of adjacent 
    coprime pairs in its sorted divisor list.
    
    Strategy: Use 15 distinct primes. To maximize the "spread" of divisors 
    (which helps break up non-coprime adjacent pairs), we keep the smallest 
    13 primes and choose the 14th and 15th primes such that the 15th is 
    as large as possible. This is achieved by minimizing the 14th prime 
    (the next prime after the 13th) and maximizing the 15th within the limit.
    """
    limit = 10**18
    
    # Generate first 13 primes
    primes = []
    candidate = 2
    while len(primes) < 13:
        is_p = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(candidate)
        candidate += 1 if candidate == 2 else 2

    P13 = 1
    for p in primes:
        P13 *= p

    # 14th prime is the next prime after the 13th
    p13 = primes[-1]
    p14 = p13 + 1
    while True:
        is_p = True
        for p in primes:
            if p * p > p14:
                break
            if p14 % p == 0:
                is_p = False
                break
        if is_p:
            break
        p14 += 1

    # Maximize 15th prime
    rem = limit // (P13 * p14)
    p15 = rem
    while p15 > p14:
        is_p = True
        for p in primes:
            if p * p > p15:
                break
            if p15 % p == 0:
                is_p = False
                break
        if is_p:
            break
        p15 -= 1
        
    # If no prime found (should not happen), fallback
    if p15 <= p14:
        return P13 * p14 # Just in case

    return P13 * p14 * p15
