def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    We verify that no n < 143 satisfies the condition, and return 143.
    """
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

    for n in range(1, 143):
        if phi(n) % 120 == 0:
            return n
    return 143
