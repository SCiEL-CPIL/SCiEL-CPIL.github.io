import math

def phi(n):
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result

def construct():
    # We know the answer is 143, but we verify it to be robust.
    # The search space is small.
    for n in range(1, 1000001):
        if phi(n) % 120 == 0:
            return n
    return -1
