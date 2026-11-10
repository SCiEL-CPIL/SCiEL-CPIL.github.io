"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    # Deadline guard, though this computation is instantaneous
    start_time = time.time()
    
    max_n = 256
    
    # Step 1: Find all primes up to max_n
    # Sieve of Eratosthenes
    is_prime = [True] * (max_n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(max_n)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_n + 1, i):
                is_prime[j] = False
                
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # Step 2: For each prime, find the maximum power <= max_n
    selected_numbers = [1] # 1 is coprime to all and distinct
    
    for p in primes:
        power = p
        next_power = p * p
        while next_power <= max_n:
            power = next_power
            next_power = power * p
        selected_numbers.append(power)
        
    # Step 3: Verify constraints (optional but good for safety)
    # Check distinct
    if len(selected_numbers) != len(set(selected_numbers)):
        # Fallback to primes + 1 if logic error (should not happen)
        return [1] + primes
        
    # Check pairwise coprime
    for i in range(len(selected_numbers)):
        for j in range(i+1, len(selected_numbers)):
            if math.gcd(selected_numbers[i], selected_numbers[j]) != 1:
                # Fallback
                return [1] + primes

    return selected_numbers
# EVOLVE-BLOCK-END
