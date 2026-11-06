import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 123457
    
    # We want to find min B such that there exist a, b with a+b=N and P(a)<=B, P(b)<=B.
    # This is equivalent to a, b being B-smooth.
    
    # Helper to get primes up to limit
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = False
        sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    # Helper to generate all B-smooth numbers <= N
    def generate_smooth(n, primes):
        smooths = set()
        
        def dfs(idx, current):
            if current > n:
                return
            if idx == len(primes):
                smooths.add(current)
                return
            p = primes[idx]
            val = current
            while val <= n:
                dfs(idx + 1, val)
                val *= p
                if time.time() > deadline:
                    return
        
        dfs(0, 1)
        return smooths

    # Try increasing B
    # Start from B=2
    # Max B to try? If we don't find it, fallback to brute force or return best found.
    # Given the previous analysis, B=11 is likely the answer.
    # We can go up to B=1000 to be safe.
    
    for B in range(2, 1001):
        if time.time() > deadline:
            break
            
        primes = get_primes(B)
        smooths = generate_smooth(N, primes)
        
        # Check for pair
        for a in smooths:
            b = N - a
            if b > 0 and b in smooths:
                return [a, b]
                
    # Fallback if not found with B < 1000
    # Use the original sieve method as a fallback, but it's slow.
    # Better fallback: just return a known good pair if we have one, or run a limited search.
    # Since the problem is fixed, we expect to find it quickly.
    
    # If we reach here, something is wrong. Return a default valid pair.
    # a=1, b=N-1
    return [1, N - 1]
