from math import gcd
from itertools import combinations
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    def check(A, B):
        sums = [a+b for a in A for b in B]
        distinct = list(set(sums))
        n = len(distinct)
        for i in range(n):
            for j in range(i+1, n):
                if gcd(distinct[i], distinct[j]) > 1:
                    return 0
        return n
    
    best_size = 0
    best_A = None
    best_B = None
    
    # Generate primes up to 128
    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i+2) == 0:
                return False
            i += 6
        return True
    
    primes = [p for p in range(2, 129) if is_prime(p)]
    
    # Search over prime sets for A and B
    prime_combos = list(combinations(primes, 3))
    
    for A in prime_combos:
        if time.time() > deadline:
            break
        for B in prime_combos:
            if time.time() > deadline:
                break
            s = check(A, B)
            if s > best_size:
                best_size = s
                best_A = list(A)
                best_B = list(B)
                if best_size == 9:
                    return [best_A, best_B]
    
    # Also search over all small values
    small_vals = list(range(1, 31))
    small_combos = list(combinations(small_vals, 3))
    
    for A in small_combos:
        if time.time() > deadline:
            break
        for B in small_combos:
            if time.time() > deadline:
                break
            s = check(A, B)
            if s > best_size:
                best_size = s
                best_A = list(A)
                best_B = list(B)
                if best_size == 9:
                    return [best_A, best_B]
    
    if best_A is None:
        return [[1, 7, 13], [4, 10, 16]]
    return [best_A, best_B]
