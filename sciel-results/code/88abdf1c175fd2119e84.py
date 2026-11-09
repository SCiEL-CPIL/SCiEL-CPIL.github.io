import math

def phi(n):
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n = n // p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result

def construct():
    # Constructive search for the minimal n such that 120 divides phi(n).
    # We iterate n from 1 upwards and compute phi(n) using prime factorization.
    # The search stops at the first valid n.
    target = 120
    for n in range(1, 1000001):
        if phi(n) % target == 0:
            return n
    return -1
