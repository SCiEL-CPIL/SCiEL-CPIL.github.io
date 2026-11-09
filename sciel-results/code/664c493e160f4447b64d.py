import math

def construct():
    N = 123457
    
    # Get primes up to N (or sqrt(N) is enough for initial M, but M might grow)
    # We will increment M. We need primes up to M.
    # Let's just check M from 2 upwards.
    
    # Helper to get primes up to limit
    def get_primes(limit):
        if limit < 2:
            return []
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    # Helper to generate all M-smooth numbers <= N
    def generate_smooth_numbers(primes, limit):
        smooths = set()
        # Recursive generation
        # To avoid deep recursion issues or excessive calls, we can use a stack or recursion.
        # Number of smooth numbers is small for small M.
        
        def dfs(idx, current_val):
            if current_val > limit:
                return
            smooths.add(current_val)
            if idx == len(primes):
                return
            p = primes[idx]
            # Multiply by p as many times as possible
            val = current_val
            while val <= limit:
                val *= p
                if val > limit:
                    break
                dfs(idx + 1, val)
                # Wait, this logic is slightly off. 
                # We should multiply by p, then recurse to next prime.
                # Then multiply by p again?
                # Standard approach:
                # For each prime p_i, we can choose exponent e_i >= 0.
                
        # Better DFS:
        def dfs2(idx, current_val):
            if current_val > limit:
                return
            smooths.add(current_val)
            if idx == len(primes):
                return
            p = primes[idx]
            # Try all powers of p
            val = current_val
            while val <= limit:
                dfs2(idx + 1, val)
                val *= p
                
        dfs2(0, 1)
        return smooths

    M = 2
    while True:
        primes = get_primes(M)
        smooths = generate_smooth_numbers(primes, N)
        
        # Check if there is a pair a, b in smooths such that a+b=N
        found = False
        for a in smooths:
            b = N - a
            if b in smooths:
                # Ensure a, b are positive
                if a > 0 and b > 0:
                    return [a, b]
        
        M += 1
        # Safety break, though M won't get very large
        if M > 1000:
             # Fallback to original method if smooth numbers are too sparse?
             # Or just continue. 1000 is large.
             pass
