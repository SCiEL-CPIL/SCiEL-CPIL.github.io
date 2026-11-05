import math

def construct():
    """
    Returns an integer n <= 10^18 that maximizes the number of adjacent 
    coprime pairs in its sorted divisor list.
    
    The optimal strategy is to use the product of the first k primes 
    (the k-th primorial) such that the product is as large as possible 
    but still <= 10^18. This maximizes the number of divisors (2^k) 
    and the frequency of coprime transitions between them.
    """
    limit = 10**18
    
    # Generate primes and compute primorial
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    n = 1
    candidate = 2
    
    while True:
        if n * candidate > limit:
            break
        n *= candidate
        candidate += 1
        if candidate % 2 == 0:
            candidate += 1
        # Skip multiples of 3
        while not is_prime(candidate):
            candidate += 2

    return n
