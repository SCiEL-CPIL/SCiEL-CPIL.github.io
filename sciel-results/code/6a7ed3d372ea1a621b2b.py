import math

def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Since the minimum n is known to be small (143), we iterate n from 1 upwards,
    compute phi(n) using trial division, and return the first n where phi(n) % 120 == 0.
    This is significantly faster than sieving up to 1000000.
    """
    def phi(n):
        if n == 1:
            return 1
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

    # The answer is expected to be small. 
    # We search up to 1000 first. If not found, we expand or use a sieve.
    # Based on analysis, 143 is the answer.
    for n in range(1, 1000):
        if phi(n) % 120 == 0:
            return n
    
    # Fallback: if for some reason it's not in 1..999, use a sieve up to 100000
    # This should never happen for this specific task, but ensures validity.
    N = 100000
    phi_sieve = list(range(N + 1))
    for i in range(2, N + 1):
        if phi_sieve[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi_sieve[j] -= phi_sieve[j] // i
    
    for n in range(1, N + 1):
        if phi_sieve[n] % 120 == 0:
            return n
            
    # Final fallback (should be unreachable)
    return 143
