import time

def construct():
    # We will use a simple trial division for small n first, 
    # or just SPF for all. SPF is safer for speed if n is larger.
    # But 143 is very small. 
    # Let's just do a simple loop with a fast phi function.
    
    def phi(n):
        if n == 1: return 1
        result = n
        x = n
        p = 2
        while p * p <= x:
            if x % p == 0:
                while x % p == 0:
                    x //= p
                result -= result // p
            p += 1
        if x > 1:
            result -= result // x
        return result

    # Iterate n
    for n in range(1, 1000001):
        if phi(n) % 120 == 0:
            return n
    return 14400 # Fallback
