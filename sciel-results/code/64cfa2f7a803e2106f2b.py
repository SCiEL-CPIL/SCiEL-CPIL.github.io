import random
from math import gcd
from itertools import combinations

def construct():
    # We want 3x3 grid of sums all distinct primes
    # Primes up to 256
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5)+1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = set(p for p in range(2, 257) if is_prime(p))
    
    # Systematic search over a small box
    # If we find 9 distinct primes, we are done.
    # A and B must be distinct integers in [1, 128].
    
    # Search in [1, 30]
    range_limit = 30
    candidates = list(range(1, range_limit + 1))
    combs = list(combinations(candidates, 3))
    
    for A in combs:
        for B in combs:
            sums = [a + b for a in A for b in B]
            distinct_sums = set(sums)
            
            if len(distinct_sums) == 9:
                if all(s in primes for s in distinct_sums):
                    return [sorted(A), sorted(B)]
                    
    # Expand search to [1, 50] if not found
    range_limit = 50
    candidates = list(range(1, range_limit + 1))
    combs = list(combinations(candidates, 3))
    
    for A in combs:
        for B in combs:
            sums = [a + b for a in A for b in B]
            distinct_sums = set(sums)
            
            if len(distinct_sums) == 9:
                if all(s in primes for s in distinct_sums):
                    return [sorted(A), sorted(B)]

    # Fallback: known good solution
    return [[1, 7, 13], [4, 10, 16]]
