"""Rewritten program for large integer sets with no two-other-product divisibility."""
# EVOLVE-BLOCK-START
import time

P = {'n': 256}

def construct():
    deadline = time.time() + 85
    n = P['n']
    
    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True
    
    primes = [x for x in range(2, n + 1) if is_prime(x)]
    S = list(primes)
    
    def can_add(a, S):
        m = len(S)
        for i in range(m):
            b = S[i]
            for j in range(i + 1, m):
                c = S[j]
                # Check a | b*c
                if b * c >= a and (b * c) % a == 0:
                    return False
                # Check b | a*c
                if a * c >= b and (a * c) % b == 0:
                    return False
                # Check c | a*b
                if a * b >= c and (a * b) % c == 0:
                    return False
        return True
    
    composites = [x for x in range(2, n + 1) if not is_prime(x)]
    for a in sorted(composites, reverse=True):
        if time.time() >= deadline:
            break
        if can_add(a, S):
            S.append(a)
    
    return S
# EVOLVE-BLOCK-END
