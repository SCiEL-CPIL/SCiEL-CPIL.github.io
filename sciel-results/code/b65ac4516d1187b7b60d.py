import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    MAX_VAL = 1000001
    # pp[i] will store the powerful part of i
    pp = [1] * (MAX_VAL + 1)
    
    # Sieve for primes up to sqrt(MAX_VAL)
    # We only need primes p where p^2 <= MAX_VAL
    limit = int(math.isqrt(MAX_VAL))
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    # Process each prime
    for p in range(2, limit + 1):
        if not is_prime[p]:
            continue
            
        # Generate powers p^2, p^3, ...
        power = p * p
        while power <= MAX_VAL:
            # For each multiple of this power, check if it contains exactly this power or more
            # We need to find the exact exponent e of p in the multiple m
            # m is a multiple of p^k. Let m = k' * p^k.
            # We divide m by p repeatedly to find total exponent.
            
            # Optimization: Instead of iterating all multiples and dividing, 
            # we can iterate multiples of p^k and check divisibility by p^(k+1).
            # If m is divisible by p^(k+1), then the exponent is at least k+1.
            # Actually, it's easier to just find the full exponent for each multiple.
            
            # Let's iterate multiples of power (which is p^k)
            # m = power, 2*power, 3*power...
            # For each m, count how many times p divides m.
            # Since m is a multiple of p^k, it has at least k factors of p.
            
            next_power = power * p
            if next_power > MAX_VAL:
                # p^k is the highest power. 
                # Any multiple of p^k that is NOT a multiple of p^(k+1) has exactly exponent k.
                # Any multiple of p^(k+1) has exponent >= k+1, which will be handled by the next iteration of the while loop (if it exists) or is already covered?
                # No, the while loop goes to higher powers.
                # If we are at the last power p^k where p^(k+1) > MAX_VAL, then:
                # Multiples of p^k:
                #   If m is NOT divisible by p^(k+1) (which is impossible since p^(k+1) > MAX_VAL >= m? No, p^(k+1) > MAX_VAL means no multiple of p^(k+1) exists in range).
                #   Wait, if p^(k+1) > MAX_VAL, then NO number <= MAX_VAL is divisible by p^(k+1).
                #   Therefore, EVERY multiple of p^k has EXACTLY exponent k.
                #   So we just multiply pp[m] by p^k for all m.
                
                # Iterate multiples of power
                for m in range(power, MAX_VAL + 1, power):
                    pp[m] *= power
                break
            else:
                # p^(k+1) exists.
                # Multiples of p^k that are NOT multiples of p^(k+1) have exponent exactly k.
                # Multiples of p^(k+1) have exponent >= k+1. They will be processed in the next iteration of the while loop.
                # So we only process m such that m % next_power != 0.
                
                # To avoid checking every multiple, we can iterate all multiples and skip those divisible by next_power?
                # Or iterate multiples of power, and check if m % next_power != 0.
                
                for m in range(power, MAX_VAL + 1, power):
                    if m % next_power != 0:
                        pp[m] *= power
                
                power = next_power

    # Find the best n in [10000, 1000000]
    best_n = 10000
    best_score = 0
    
    # Use local variables for speed
    pp_local = pp
    for n in range(10000, 1000001):
        score = pp_local[n] * pp_local[n+1]
        if score > best_score:
            best_score = score
            best_n = n
            
        # Time check (unlikely to be needed for 1e6 iterations, but safe)
        if time.time() > deadline:
            break
            
    return best_n
