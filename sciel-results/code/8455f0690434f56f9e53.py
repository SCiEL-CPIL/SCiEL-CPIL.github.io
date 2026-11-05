import time
import math

def construct():
    deadline = time.time() + 85
    
    TARGET = 123457
    
    # Generate primes up to TARGET using sieve
    def sieve(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(TARGET)
    
    # Incrementally build the smooth set as we add primes one by one
    smooths = {1}
    
    for k in primes:
        if time.time() > deadline:
            break
        
        # Add all numbers formed by multiplying existing smooths by powers of k
        new = set()
        for s in smooths:
            v = s
            while v * k <= TARGET:
                v *= k
                new.add(v)
        smooths.update(new)
        
        # Check if any two smooth numbers sum to TARGET
        found = None
        for a in smooths:
            b = TARGET - a
            if b >= 1 and b in smooths:
                found = (a, b)
                break
        
        if found:
            return list(found)
    
    # Last resort
    return [1, 123456]
