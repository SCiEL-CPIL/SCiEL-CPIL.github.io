import math
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a,b) > 1 for all distinct pairs. 
    Maximizes cardinality.
    """
    N = 255
    deadline = time.time() + 55.0  # Safety margin before 60s limit
    
    # Build adjacency: two numbers are adjacent if gcd > 1
    # We need a clique containing 255
    
    # First, get a good initial solution using the prime-based approach
    def get_prime_based_set(p):
        if N % p == 0:
            return [x for x in range(1, N + 1) if x % p == 0]
        else:
            candidates = [x for x in range(1, N + 1) if x % p == 0 and math.gcd(x, N) > 1]
            candidates.append(N)
            return candidates
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(N)
    
    best_set = [255]
    best_size = 1
    
    for p in primes:
        if time.time() > deadline:
            break
        candidates = get_prime_based_set(p)
        if len(candidates) > best_size:
            best_size = len(candidates)
            best_set = candidates
    
    # Now try to improve via local search: add elements one at a time if compatible
    # Convert to set for O(1) lookup
    best_set_set = set(best_set)
    
    # Precompute which numbers are compatible with all elements in best_set
    # A number x can be added if gcd(x, y) > 1 for all y in best_set
    # and x != 255 (already in set) and x not already in set
    
    # Try adding elements greedily
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        for x in range(1, N + 1):
            if x in best_set_set:
                continue
            # Check if x is compatible with all elements in best_set
            compatible = True
            for y in best_set:
                if math.gcd(x, y) == 1:
                    compatible = False
                    break
            if compatible:
                best_set.append(x)
                best_set_set.add(x)
                best_size += 1
                improved = True
    
    return best_set
