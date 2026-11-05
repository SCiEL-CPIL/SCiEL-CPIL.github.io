import random

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    N = 1000024  # Max value of n+i is 1000000 + 24
    
    # Sieve to compute the 23-smooth part of each number
    smooth = [1] * (N + 1)
    for p in primes:
        for j in range(p, N + 1, p):
            x = j
            while x % p == 0:
                smooth[j] *= p
                x //= p
                
    best_n = 1000
    best_count = float('inf')
    
    # Slide a window of size 24 over the range [1001, 1000000+24]
    # n ranges from 1000 to 1000000
    # Window covers smooth[n+1] to smooth[n+24]
    for n in range(1000, 1000001):
        window = smooth[n+1:n+25]
        count = len(set(window))
        if count < best_count:
            best_count = count
            best_n = n
            
    return best_n
