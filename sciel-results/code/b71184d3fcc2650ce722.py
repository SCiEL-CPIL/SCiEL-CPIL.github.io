"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import math

def construct():
    """
    Returns a list of distinct integers from 1..256 such that every pair is coprime,
    maximizing the sum.
    """
    N = 256
    
    # Step 1: Find all primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    if N >= 1:
        is_prime[1] = False
        
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Step 2: For each prime, find the largest power p^k <= N
    prime_to_power = {}
    for p in primes:
        power = p
        next_power = p * p
        while next_power <= N:
            power = next_power
            next_power *= p
        prime_to_power[p] = power
        
    # Initial set: 1 and all largest prime powers
    current_set = [1]
    current_set.extend(prime_to_power.values())
    
    # Helper to get prime factors
    def get_prime_factors(n):
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.append(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    # Local Search: Try to replace prime powers with composites if beneficial
    # Iterate composites in descending order to prioritize larger swaps
    composites = [n for n in range(N, 1, -1) if not is_prime[n] and n != 1]
    
    for n in composites:
        factors = get_prime_factors(n)
        if not factors:
            continue
            
        # Check if all corresponding prime powers are in the current set
        # And calculate the sum of those prime powers
        powers_to_remove = []
        sum_powers = 0
        valid_swap = True
        
        for p in factors:
            pp = prime_to_power[p]
            if pp in current_set:
                powers_to_remove.append(pp)
                sum_powers += pp
            else:
                # If a prime power is not in the set, we can't do this specific swap
                # (It might have been swapped out already, or it wasn't the largest power chosen? 
                # Actually, our set always contains the largest power for each prime UNLESS swapped out.
                # If swapped out, we don't want to put it back via this mechanism easily.
                # So we skip this composite if any of its prime powers are missing.)
                valid_swap = False
                break
                
        if not valid_swap:
            continue
            
        gain = n - sum_powers
        if gain <= 0:
            continue
            
        # Check if n is coprime to all OTHER elements in the set
        # The set consists of powers_to_remove + other_elements
        # We remove powers_to_remove, so we check against the rest
        # We also need to check if n is coprime to 1 (always true)
        
        # Create a set of elements to check against (excluding those to be removed)
        # Note: current_set is a list. 
        elements_to_check = [x for x in current_set if x not in powers_to_remove]
        
        is_coprime_to_rest = True
        for x in elements_to_check:
            if math.gcd(n, x) != 1:
                is_coprime_to_rest = False
                break
                
        if is_coprime_to_rest:
            # Perform swap
            # Remove powers_to_remove
            for pwr in powers_to_remove:
                current_set.remove(pwr)
            # Add n
            current_set.append(n)
            
    return current_set

# EVOLVE-BLOCK-END
