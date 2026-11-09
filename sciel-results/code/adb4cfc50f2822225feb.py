import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    limit = 10**18
    
    # First 15 primes
    # 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    n = 1
    for p in primes:
        if n * p > limit:
            break
        n *= p
        
    # n is the product of the first 15 primes
    # 614889782588491410
    return n
