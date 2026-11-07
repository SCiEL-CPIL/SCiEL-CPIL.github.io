"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 256
    
    # Sieve of Eratosthenes
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    composites = [x for x in range(2, N + 1) if not is_prime[x]]
    
    # Start with all primes (known valid)
    S = set(primes)
    
    def can_add(x, S):
        """Check if x can be added to set S while maintaining validity."""
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                # Triple (x, a, b): x must not divide a*b
                if (a * b) % x == 0:
                    return False
                # Triple (a, x, b): a must not divide x*b
                if (x * b) % a == 0:
                    return False
                # Triple (b, x, a): b must not divide x*a
                if (x * a) % b == 0:
                    return False
        return True
    
    # Greedily add composites
    for x in composites:
        if time.time() > deadline:
            break
        if can_add(x, S):
            S.add(x)
    
    # Local search: try swaps to potentially improve
    improved = True
    swap_count = 0
    max_swaps = 10000
    while improved and swap_count < max_swaps and time.time() < deadline - 5:
        improved = False
        S_list = sorted(S)
        for x_out in S_list:
            if time.time() > deadline - 5 or swap_count >= max_swaps:
                break
            for x_in in composites:
                if x_in in S:
                    continue
                new_S = S - {x_out} | {x_in}
                if can_add(x_in, new_S):
                    S = new_S
                    improved = True
                    swap_count += 1
                    break
    
    result = [int(x) for x in sorted(S)]
    return result

# EVOLVE-BLOCK-END
