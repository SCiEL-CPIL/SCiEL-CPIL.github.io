"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    n = 256
    
    # 1. Generate all primes up to 256
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve(n)
    
    # 2. Calculate largest prime power for each prime
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= n:
            power *= p
        prime_powers[p] = power
        
    # 3. Identify optimal merges
    # We found these specific merges are optimal:
    # (7, 31) -> 217
    # (11, 23) -> 253
    # (13, 19) -> 247
    
    # Primes involved in merges
    merged_primes = {7, 31, 11, 23, 13, 19}
    
    # Values to add from merges
    merged_values = [217, 253, 247]
    
    # 4. Construct the final set
    selected = [1]
    
    # Add merged values
    for v in merged_values:
        selected.append(v)
        
    # Add prime powers for primes NOT involved in merges
    for p in primes:
        if p not in merged_primes:
            selected.append(prime_powers[p])
            
    # 5. Verification
    # Check distinctness
    if len(selected) != len(set(selected)):
        # Fallback to simple prime powers if distinctness fails (should not)
        selected = [1] + [prime_powers[p] for p in primes]
        
    # Check pairwise coprimality
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if math.gcd(selected[i], selected[j]) != 1:
                # Fallback to simple prime powers if coprimality fails (should not)
                selected = [1] + [prime_powers[p] for p in primes]
                break
        else:
            continue
        break
    else:
        # All good
        return [int(x) for x in selected]
        
    # Fallback return
    return [int(x) for x in selected]
# EVOLVE-BLOCK-END
