"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math
    
    # Generate primes less than 512
    is_prime = [True] * 512
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(511)) + 1):
        if is_prime[i]:
            for j in range(i*i, 512, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, 512) if is_prime[i]]
    
    # Initialize with all primes
    current_set = set(primes)
    
    # Helper to get prime factors
    def get_prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.add(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    # Iterate through composites in descending order of value (to prioritize high weights)
    for c in range(511, 3, -1):
        if c in current_set:
            continue # It's a prime, already handled
            
        # Get prime factors of c
        p_factors = get_prime_factors(c)
        
        # Check if all prime factors are currently in the set
        if not p_factors.issubset(current_set):
            continue
            
        # Check if c is coprime to all other elements in the set (excluding its factors)
        # Since the rest of the set consists of primes (or other composites we might have added),
        # we just need gcd(c, s) == 1 for all s in current_set - p_factors
        valid = True
        for s in current_set:
            if s in p_factors:
                continue
            if math.gcd(c, s) != 1:
                valid = False
                break
                
        if not valid:
            continue
            
        # Calculate gain
        # Gain = weight(c) - sum(weight(p) for p in p_factors)
        gain = 1.0 / (512 - c) - sum(1.0 / p for p in p_factors)
        
        if gain > 1e-9: # Small epsilon to avoid floating point issues, though exact rational is better
            # Perform swap
            current_set -= p_factors
            current_set.add(c)
            
    return list(current_set)
# EVOLVE-BLOCK-END
