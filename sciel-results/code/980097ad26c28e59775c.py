import math

def construct():
    # The task is to maximize the number of adjacent pairs of divisors d_i, d_{i+1}
    # in the sorted list of divisors of n such that gcd(d_i, d_{i+1}) = 1.
    # 
    # Analysis:
    # 1. To maximize the number of coprime adjacent pairs, we want the divisors to be
    #    "spread out" in terms of their prime factorization.
    # 2. Square-free numbers with many distinct small prime factors are optimal because:
    #    - They allow for the maximum number of distinct prime factors k within the limit 10^18.
    #    - The smallest primes (2, 3, 5, ...) ensure the product stays as small as possible,
    #      allowing for more primes (higher k).
    #    - In a square-free number, each prime appears in exactly half the divisors.
    #      The sorted order of divisors tends to interleave subsets with disjoint supports
    #      more effectively than numbers with high powers, which create clusters of
    #      multiples sharing common factors.
    #
    # 3. We find the largest k such that the product of the first k primes (primorial)
    #    is <= 10^18.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
    n = 1
    for p in primes:
        if n * p <= 10**18:
            n *= p
        else:
            break
            
    return n
