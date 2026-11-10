import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    N = 256
    
    # 1. Generate primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # 2. For each prime, find max power <= N
    # p_max_map[p] = largest p^k <= N
    p_max_val = {}
    for p in primes:
        max_pow = p
        while max_pow * p <= N:
            max_pow *= p
        p_max_val[p] = max_pow

    # Initial candidate set: 1 and all max powers
    # Note: If we pick max power for p, we cannot pick any other multiple of p.
    # The set of max powers is pairwise coprime.
    current_set = {1}
    for p in primes:
        current_set.add(p_max_val[p])
        
    current_sum = sum(current_set)
    
    # 3. Local Search: Try to replace two max powers with their product
    # If p * q <= N and p * q > p_max_val[p] + p_max_val[q]
    # We replace p_max_val[p] and p_max_val[q] with p * q
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate over pairs of primes
        for i in range(len(primes)):
            if time.time() > deadline:
                break
            p = primes[i]
            for j in range(i + 1, len(primes)):
                q = primes[j]
                prod = p * q
                if prod > N:
                    continue # Product too large, and higher primes will be larger
                # Check if we can improve
                # We are currently holding p_max_val[p] and p_max_val[q]
                # We can replace them with prod
                # This is valid because prod uses only primes p and q,
                # and no other number in the set uses p or q (since we only have max powers for other primes,
                # and max powers of other primes are coprime to p and q).
                val_p = p_max_val[p]
                val_q = p_max_val[q]
                if prod > val_p + val_q:
                    # Perform replacement
                    current_set.remove(val_p)
                    current_set.remove(val_q)
                    current_set.add(prod)
                    current_sum += prod - val_p - val_q
                    improved = True
                    # Update the "effective" values for p and q?
                    # Actually, p and q are now "consumed" by prod.
                    # We should mark them as such or handle it carefully.
                    # Since we are iterating, let's just restart or be careful.
                    # Simplest: restart the loop or mark as done.
                    # Given small N, restart is safe.
                    break
            if improved:
                break
                
    # Convert to list of ints
    result = [int(x) for x in current_set]
    return result
