"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import heapq
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    
    N = 123457
    
    # Generate primes up to N using a simple sieve
    # This is fast enough for N=123457
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
                
    # Iterate through primes as potential max prime factor K
    # The first K for which we find a pair a+b=N with a,b K-smooth
    # will give the minimum possible max(P(a), P(b))
    for K in primes:
        if time.time() > deadline:
            break
            
        # Generate all K-smooth numbers <= N
        # Using a min-heap to generate in sorted order (though we use a set for lookup)
        # We can also just generate the set directly
        smooths = set([1])
        heap = [1]
        
        while heap:
            x = heapq.heappop(heap)
            if x > N:
                continue
            for p in primes:
                if p > K:
                    break
                nx = x * p
                if nx > N:
                    break
                if nx not in smooths:
                    smooths.add(nx)
                    heapq.heappush(heap, nx)
        
        # Check if any two numbers in smooths sum to N
        # a + b = N => b = N - a
        # Since we need a, b positive, a must be in [1, N-1]
        found_pair = None
        for a in smooths:
            b = N - a
            if b in smooths:
                # Both are positive since a <= N-1 (as 1 is in smooths, N-1 must be in smooths for b=1)
                # Actually if a=N, b=0, but 0 is not in smooths (we start with 1)
                # So b >= 1 is guaranteed if b is in smooths
                found_pair = [a, b]
                break
        
        if found_pair:
            return found_pair
            
    # Fallback: If no solution found within time or primes exhausted (should not happen)
    # Return a valid pair, e.g., 1 and N-1
    # P(1)=1, P(N-1) is whatever it is
    return [1, N - 1]
# EVOLVE-BLOCK-END
