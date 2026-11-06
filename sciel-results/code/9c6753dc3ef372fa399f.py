import math

def construct():
    """
    Find minimum n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses direct search with on-the-fly totient calculation.
    """
    target = 120
    max_n = 1000000
    
    def totient(n):
        result = n
        if n % 2 == 0:
            result -= result // 2
            while n % 2 == 0:
                n //= 2
        p = 3
        while p * p <= n:
            if n % p == 0:
                result -= result // p
                while n % p == 0:
                    n //= p
            p += 2
        if n > 1:
            result -= result // n
        return result

    for n in range(1, max_n + 1):
        if totient(n) % target == 0:
            return n
            
    return 143
