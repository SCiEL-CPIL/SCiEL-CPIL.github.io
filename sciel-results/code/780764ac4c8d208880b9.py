import time
import heapq

def construct():
    deadline = time.time() + 85.0
    TARGET = 123457
    
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
    
    def generate_smooth_numbers(primes, limit):
        """Generate all numbers <= limit whose prime factors are all in primes."""
        if not primes:
            return set()
        smooth = set([1])
        for p in primes:
            new_smooth = set(smooth)
            for x in list(smooth):
                y = x * p
                while y <= limit:
                    new_smooth.add(y)
                    y *= p
            smooth = new_smooth
        return {x for x in smooth if 1 <= x <= limit}
    
    # Precompute primes
    all_primes = []
    n = 2
    while len(all_primes) < 50:
        if is_prime(n):
            all_primes.append(n)
        n += 1
    
    # Try increasing sets of primes
    for num_primes in range(1, len(all_primes) + 1):
        if time.time() > deadline - 5:
            break
        primes = all_primes[:num_primes]
        
        # Generate all smooth numbers
        smooth = generate_smooth_numbers(primes, TARGET)
        
        # Check if any two smooth numbers sum to TARGET
        for a in smooth:
            if time.time() > deadline - 3:
                break
            b = TARGET - a
            if 1 <= b <= TARGET and b in smooth:
                return [a, b]
    
    # Fallback
    return [1, TARGET - 1]
