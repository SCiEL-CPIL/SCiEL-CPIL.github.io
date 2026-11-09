import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin out of 60s
    
    N = 123457
    
    # Sieve of Eratosthenes to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # A contains all numbers <= N whose prime factors are in the set of primes processed so far
    # Start with 1 (empty product)
    A = {1}
    
    for p in primes:
        if time.time() > deadline:
            # Fallback: return a valid pair, even if not optimal
            # 1 and N-1 is always valid, but P(N-1) might be large.
            # Just return the best found? We didn't save best.
            # Let's just return [1, N-1] to be safe, though unlikely to be reached.
            return [1, N - 1]
            
        # Generate new smooth numbers involving p
        # We want to add numbers of form x * p^k where x in A and x * p^k <= N
        new_numbers = set()
        for x in A:
            val = x
            while val <= N:
                new_numbers.add(val)
                if val > N // p:
                    break
                val *= p
        
        A.update(new_numbers)
        
        # Check if any a in A has N-a in A
        # Optimization: only check a <= N/2 to avoid duplicate pairs, but set lookup is O(1)
        # We just need to find ONE pair.
        found = False
        for a in A:
            b = N - a
            if b > 0 and b in A:
                return [a, b]
        
        # If no pair found, continue to next prime
        # Note: A is growing. The check loop iterates over A.
        # Size of A (smooth numbers) grows slowly.
        # For p=2, A has powers of 2.
        # For p=3, A has 2^i * 3^j.
        # It is expected to find a solution for small p (e.g., 5 or 7).

    # If loop finishes without finding a pair (should not happen for N=123457)
    return [1, N - 1]
