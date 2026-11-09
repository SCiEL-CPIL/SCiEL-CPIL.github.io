"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """Return a list of distinct ints from 1..256, pairwise coprime, maximizing sum."""
    deadline = time.time() + 50  # safety margin before 60s kill
    N = 256
    
    # Find all primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, math.isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # For each prime, find the largest power p^k <= N
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power
        
    result = [1] + list(prime_powers.values())
    
    # Apply specific improvements found by analysis
    # 1. Replace {121, 23} with {253} (Gain 109)
    if 121 in result and 23 in result:
        result.remove(121)
        result.remove(23)
        result.append(253)
        
    # 2. Replace {49, 31} with {217} (Gain 137)
    if 49 in result and 31 in result:
        result.remove(49)
        result.remove(31)
        result.append(217)
        
    # 3. Replace {125, 47} with {235} (Gain 63)
    if 125 in result and 47 in result:
        result.remove(125)
        result.remove(47)
        result.append(235)
        
    # 4. Replace {81, 83} with {249} (Gain 85)
    if 81 in result and 83 in result:
        result.remove(81)
        result.remove(83)
        result.append(249)
        
    # 5. Replace {169, 19} with {247} (Gain 59)
    if 169 in result and 19 in result:
        result.remove(169)
        result.remove(19)
        result.append(247)
        
    # Verify pairwise coprime (sanity check)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                # Fallback: return just prime powers for safety
                return [1] + list(prime_powers.values())
    
    return result
# EVOLVE-BLOCK-END
