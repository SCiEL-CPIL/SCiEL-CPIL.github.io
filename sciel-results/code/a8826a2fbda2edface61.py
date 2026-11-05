import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Helper: LCM calculation
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Helper: count distinct prime factors
    def num_distinct_primes(x):
        count = 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                count += 1
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            count += 1
        return count

    # Generate prime powers
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(n)
    prime_powers = set()
    for p in primes:
        power = p
        while power <= n:
            prime_powers.add(power)
            power *= p
            
    # Core set: prime powers + 1
    current_set = set(prime_powers | {1})
    
    # Greedy addition of remaining numbers
    remaining = [x for x in range(1, n + 1) if x not in current_set]
    remaining.sort(key=lambda x: (num_distinct_primes(x), x))
    
    # Check if adding x creates a bad triple
    for x in remaining:
        if time.time() > deadline:
            break
            
        valid = True
        groups = {}
        for a in current_set:
            L = lcm(x, a)
            if L not in groups:
                groups[L] = []
            groups[L].append(a)
            
        for L, elements in groups.items():
            if len(elements) < 2:
                continue
            for i in range(len(elements)):
                for j in range(i + 1, len(elements)):
                    if lcm(elements[i], elements[j]) == L:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            current_set.add(x)
    
    # Polish: try to add back removed nodes if possible
    removed = [x for x in numbers if x not in current_set]
    removed.sort(key=lambda x: (num_distinct_primes(x), x))
    
    for x in removed:
        if time.time() > deadline - 2.0:
            break
        valid = True
        groups = {}
        for a in current_set:
            L = lcm(x, a)
            if L not in groups:
                groups[L] = []
            groups[L].append(a)
            
        for L, elements in groups.items():
            if len(elements) < 2:
                continue
            for i in range(len(elements)):
                for j in range(i + 1, len(elements)):
                    if lcm(elements[i], elements[j]) == L:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            current_set.add(x)
    
    return list(current_set)
