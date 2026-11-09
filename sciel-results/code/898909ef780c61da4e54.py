import math

def construct():
    n = 255
    
    # Find prime factors of n
    factors = []
    d = 2
    temp_n = n
    while d * d <= temp_n:
        if temp_n % d == 0:
            factors.append(d)
            while temp_n % d == 0:
                temp_n //= d
        d += 1
    if temp_n > 1:
        factors.append(temp_n)
    
    # For each prime factor, calculate the size of the set of multiples
    best_set = []
    max_size = 0
    
    for p in factors:
        # The set of multiples of p in [1, n]
        count = n // p
        if count > max_size:
            max_size = count
            # Construct the set
            best_set = list(range(p, n + 1, p))
            
    # Ensure n is in the set (it is, since p divides n)
    # The set is non-empty and contains n.
    # All pairs have gcd >= p > 1.
    
    return best_set
