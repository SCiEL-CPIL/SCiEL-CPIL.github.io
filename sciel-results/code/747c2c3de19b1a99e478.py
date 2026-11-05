import time
import math

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
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
    
    # Generate all K-smooth numbers up to TARGET
    def smooth_numbers(k, limit):
        """Generate all numbers ≤ limit whose prime factors are all ≤ k."""
        # Find primes ≤ k
        ps = [p for p in primes if p <= k]
        result = set([1])
        # Use a priority queue or iterative approach
        # Better: use a set and multiply
        smooths = {1}
        for p in ps:
            new = set()
            for s in smooths:
                v = s
                while v * p <= limit:
                    v *= p
                    new.add(v)
            smooths |= new
        return smooths
    
    # Binary search on K
    # Lower bound: 2 (only powers of 2)
    # Upper bound: 643 (seed value)
    lo, hi = 2, 643
    best_pair = None
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        smooths = smooth_numbers(mid, TARGET)
        
        # Check if any two smooth numbers sum to TARGET
        found = None
        for a in smooths:
            b = TARGET - a
            if b >= 1 and b in smooths:
                found = (a, b)
                break
        
        if found:
            best_pair = found
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_pair is None:
        # Fallback: try K=643 explicitly
        smooths = smooth_numbers(643, TARGET)
        for a in smooths:
            b = TARGET - a
            if b >= 1 and b in smooths:
                best_pair = (a, b)
                break
    
    if best_pair is None:
        # Last resort fallback
        return [1, 123456]
    
    return list(best_pair)
