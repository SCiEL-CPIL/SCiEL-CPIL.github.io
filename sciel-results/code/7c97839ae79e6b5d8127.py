import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    
    # 1. Generate primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # 2. Map each prime to its largest power <= N
    prime_powers = {}
    for p in primes:
        val = p
        while val * p <= N:
            val *= p
        prime_powers[p] = val
        
    # 3. Initial set: 1 and all prime powers
    # We will represent the set as a dictionary mapping prime -> value included
    # If a prime is swapped out, it won't be in this dict, or its value will be replaced.
    # Actually, it's easier to start with the list of values and modify it.
    
    initial_values = [1]
    for p in primes:
        initial_values.append(prime_powers[p])
        
    # 4. Identify potential swaps
    # A swap replaces P_p and P_q with p*q
    # Conditions: p*q <= N and p*q > P_p + P_q
    # We model this as a bipartite graph between "small" primes (those with P_p > p) and "large" primes.
    # Actually, any prime p can be in a swap. But as analyzed, only p in {5,7,11,13} have beneficial swaps.
    # Let's just generate all valid swaps.
    
    swaps = []
    for i, p in enumerate(primes):
        for j in range(i + 1, len(primes)):
            q = primes[j]
            prod = p * q
            if prod > N:
                break # Since primes are sorted, further q will be larger
            pp = prime_powers[p]
            pq = prime_powers[q]
            gain = prod - (pp + pq)
            if gain > 0:
                swaps.append((p, q, prod, gain))
                
    # 5. Select optimal set of swaps
    # This is a Max Weight Matching problem on a bipartite graph (or general graph, but effectively bipartite here).
    # Given the small number of candidates, we can use a simple greedy with conflict resolution or brute force if small enough.
    # Let's filter swaps to only those involving "small" primes (p < 17) as left side?
    # Actually, let's just try to pick swaps greedily by gain, ensuring no prime is reused.
    
    # Sort swaps by gain descending
    swaps.sort(key=lambda x: x[3], reverse=True)
    
    selected_swaps = []
    used_primes = set()
    
    for p, q, prod, gain in swaps:
        if p not in used_primes and q not in used_primes:
            selected_swaps.append((p, q, prod, gain))
            used_primes.add(p)
            used_primes.add(q)
            
    # 6. Construct final list
    final_set = set()
    
    # Add 1
    final_set.add(1)
    
    # Add prime powers for primes NOT in any selected swap
    for p in primes:
        if p not in used_primes:
            final_set.add(prime_powers[p])
            
    # Add the products for selected swaps
    for p, q, prod, gain in selected_swaps:
        final_set.add(prod)
        
    # Convert to list of ints
    result = list(final_set)
    
    # Verification
    # Check distinct
    if len(result) != len(set(result)):
        # Should not happen
        pass
        
    # Check coprime
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                # Invalid, fallback to initial prime powers
                return [1] + [prime_powers[p] for p in primes]
                
    return result
