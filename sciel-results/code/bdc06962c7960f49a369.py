"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s limit
    
    n = 256
    
    # Sieve of Eratosthenes for primes up to n
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    # For each prime p <= n, include the largest power of p that is <= n
    result = [1]
    for p in range(2, n + 1):
        if time.time() > deadline:
            break
        
        if not is_prime[p]:
            continue
        
        # Find largest power of p <= n
        power = p
        while power * p <= n:
            power *= p
        
        result.append(power)
    
    # Local search: try to improve by swapping elements
    # For each element in result, try replacing it with a larger number
    # that is coprime to all other elements
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        for i in range(len(result)):
            if time.time() > deadline:
                break
            
            current = result[i]
            # Try all numbers from n down to current+1
            for candidate in range(n, current, -1):
                if candidate == current:
                    continue
                # Check if candidate is coprime to all other elements
                valid = True
                for j in range(len(result)):
                    if j == i:
                        continue
                    if gcd(candidate, result[j]) != 1:
                        valid = False
                        break
                if valid:
                    # Check candidate is not already in result
                    if candidate not in result:
                        result[i] = candidate
                        improved = True
                        break
    
    # Ensure all are Python ints and return as list
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
