import time
import math
import heapq

def construct():
    deadline = time.time() + 85
    TARGET = 123457
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # Primes up to TARGET//2 are sufficient
    primes_all = sieve(TARGET // 2)
    
    # Incremental smooth number generation
    # smooth_set contains all numbers <= TARGET that are p-smooth for all p <= current_prime
    smooth_set = set([1])
    
    def add_smooth_numbers_for_prime(p):
        """Add all numbers that become smooth when p is allowed."""
        prev_smooth = list(smooth_set)
        new_queue = []
        
        # Generate new numbers by multiplying existing smooth numbers by powers of p
        for x in prev_smooth:
            y = x * p
            if y <= TARGET and y not in smooth_set:
                smooth_set.add(y)
                new_queue.append(y)
        
        # Propagate to higher powers of p
        while new_queue:
            x = new_queue.pop()
            y = x * p
            if y <= TARGET and y not in smooth_set:
                smooth_set.add(y)
                new_queue.append(y)
        
        # Check for pairs a, b such that a + b = TARGET and both are in smooth_set
        # We can optimize this by only checking a <= TARGET // 2 to avoid duplicates, 
        # but since we just need to return ANY pair, checking all is fine.
        # However, to be safe and fast, we can break early if found.
        for a in smooth_set:
            if a > TARGET:
                continue
            b = TARGET - a
            if b in smooth_set:
                return [a, b]
        return None

    # Initialize with prime 2
    result = add_smooth_numbers_for_prime(2)
    if result:
        return result

    # Iterate through remaining primes
    for p in primes_all[1:]:
        if time.time() > deadline - 1:
            break
        if p > TARGET // 2:
            break
            
        result = add_smooth_numbers_for_prime(p)
        if result:
            return result
            
    # Fallback if no pair found (should not happen)
    return [1, TARGET]

