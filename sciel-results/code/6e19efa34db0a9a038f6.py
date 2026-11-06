import math
import time

def construct():
    S = 123457
    start_time = time.time()
    
    # Sieve to get primes up to S
    # We only need primes up to S, but we will iterate them in order.
    # We can generate primes on the fly or use a simple sieve.
    limit = S
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1: is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    def gen_smooth(primes_list, limit_val):
        """Generate all numbers <= limit_val with prime factors in primes_list."""
        results = set()
        n_primes = len(primes_list)
        
        # Recursive generation
        # To avoid deep recursion or stack overflow, we can use a stack or iterative approach.
        # But depth is small (max exponent of 2 is ~17).
        def dfs(idx, current):
            if current > limit_val:
                return
            results.add(current)
            for i in range(idx, n_primes):
                p = primes_list[i]
                next_val = current * p
                if next_val > limit_val:
                    break
                dfs(i, next_val)
        
        dfs(0, 1)
        return results

    # Iterate through primes in increasing order
    # We want the smallest K (largest prime factor allowed) such that a solution exists.
    # K is the largest prime in the list of allowed primes.
    # So we add primes one by one.
    
    allowed_primes = []
    
    for p in primes:
        allowed_primes.append(p)
        
        # Check time limit
        if time.time() - start_time > 50: # Safety margin
             break
             
        smooth_set = gen_smooth(allowed_primes, S)
        
        # Check if there exists a, b in smooth_set such that a+b=S
        # a and b must be positive.
        # Since S is odd, a != b.
        found = False
        best_pair = None
        
        # Iterate through the set
        # To make it deterministic and potentially find smaller a? 
        # The task doesn't specify which pair to return if multiple exist with same max P.
        # "Minimize max(P(a), P(b))". Any pair achieving the minimum is fine.
        for a in smooth_set:
            b = S - a
            if b in smooth_set:
                # Both a and b are in the set, so their prime factors are in allowed_primes.
                # Thus max(P(a), P(b)) <= p.
                found = True
                best_pair = [a, b]
                break
                
        if found:
            return best_pair

    # If no solution found within time or primes, fallback to seed or best found
    # The seed is [1, 123456]. P(123456) = 151.
    # If we didn't find anything better, return the seed.
    # But we likely will find something.
    return [1, 123456]
