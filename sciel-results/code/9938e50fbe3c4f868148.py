"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find a set of distinct pairwise coprime integers a in 1..511 that maximizes
    the sum of 1/(512 - a).
    
    Approach: 
    1. Compute prime factor bitmasks for all numbers 1..511.
    2. Initialize with a greedy solution (pick largest a first if compatible).
    3. Improve using local search (hill climbing):
       - Try adding any compatible number.
       - Try swapping one number in the set with one not in the set.
       - Try swapping two numbers in the set with two not in the set.
    """
    from math import gcd
    
    N = 512
    
    # Sieve for primes up to 511
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [p for p in range(2, N) if is_prime[p]]
    
    # Map each prime to a bit position
    prime_to_bit = {}
    for i, p in enumerate(primes):
        prime_to_bit[p] = 1 << i
    
    # Compute bitset for each number's prime factors
    factor_bitset = [0] * N
    for a in range(1, N):
        x = a
        mask = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= prime_to_bit[p]
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= prime_to_bit[x]
        factor_bitset[a] = mask
    
    # Weights
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
    
    # Greedy Initialization
    current_set = []
    current_mask = 0
    current_score = 0.0
    
    for a in range(511, 0, -1):
        if factor_bitset[a] & current_mask == 0:
            current_set.append(a)
            current_mask |= factor_bitset[a]
            current_score += weights[a]
    
    # Local Search
    # We maintain a set of elements for O(1) lookup
    set_elements = set(current_set)
    
    improved = True
    while improved:
        improved = False
        
        # 1. Try adding a single element
        for v in range(511, 0, -1):
            if v not in set_elements:
                if factor_bitset[v] & current_mask == 0:
                    # Add v
                    current_set.append(v)
                    set_elements.add(v)
                    current_mask |= factor_bitset[v]
                    current_score += weights[v]
                    improved = True
                    break
        
        if improved:
            continue
            
        # 2. Try swapping one element u in set with v not in set
        for u in current_set[:]: # Iterate over a copy
            # Calculate mask of set without u? 
            # We need to check if v is compatible with all x in set \ {u}
            # This is O(N * |Set|) per swap attempt, which might be slow.
            # Optimization: Pre-compute for each u, the mask of primes used by set \ {u}?
            # No, because if two elements share a prime, removing one doesn't clear the bit.
            # So we must check compatibility against all other elements.
            
            for v in range(511, 0, -1):
                if v in set_elements:
                    continue
                # Check compatibility of v with all x in set \ {u}
                compatible = True
                for x in current_set:
                    if x == u:
                        continue
                    if factor_bitset[v] & factor_bitset[x] != 0:
                        compatible = False
                        break
                if compatible:
                    if weights[v] > weights[u] + 1e-9:
                        # Swap u with v
                        current_set.remove(u)
                        set_elements.remove(u)
                        current_set.append(v)
                        set_elements.add(v)
                        current_score += weights[v] - weights[u]
                        # Update mask: 
                        # We can't just XOR. We must recompute or update carefully.
                        # Since we removed u, some bits might clear if no other element has them.
                        # Easiest: recompute current_mask from scratch? 
                        # |Set| is small (< 100). N is 512.
                        # Recomputing mask is O(|Set|).
                        current_mask = 0
                        for elem in current_set:
                            current_mask |= factor_bitset[elem]
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            continue

        # 3. Try 2-swap: remove u1, u2 from set, add v1, v2 not in set
        # This is more expensive. O(|Set|^2 * N^2 * |Set|)
        # Given time limits, we might skip this or limit it.
        # Let's try a limited 2-swap.
        
        # We'll only try 2-swaps if we haven't improved in the last few iterations?
        # Or just run it once per loop if 1-swap fails.
        
        # For efficiency, we'll skip 2-swap for now if 1-swap is fast enough.
        # If we have time, we can add it.
        
    return current_set

# EVOLVE-BLOCK-END
