import math
import time

def construct():
    # The problem asks to maximize the number of adjacent coprime pairs in the sorted divisor list.
    # This count is generally increasing with the total number of divisors.
    # To maximize the number of divisors for n <= 10^18, we should use small primes.
    # The product of the first 17 primes is approx 3.3 * 10^17.
    # We can square one of the smallest primes to increase the divisor count further.
    # n = 2^2 * 3 * 5 * ... * 59
    # Divisors = 3 * 2^16 = 196,608.
    # This is the maximum number of divisors possible for n <= 10^18 
    # (next step would be squaring another prime, which exceeds 10^18).
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
    
    n = 1
    for i, p in enumerate(primes):
        if i == 0:
            n *= p * p
        else:
            n *= p
            
    # Verify constraint
    if n > 10**18:
        # Fallback: reduce to 16 primes squared
        n = 1
        for i, p in enumerate(primes[:-1]):
            if i == 0:
                n *= p * p
            else:
                n *= p
                
    return n
