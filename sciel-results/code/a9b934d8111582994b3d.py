"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    N = 1000000  # max n
    LO = 10000   # min n
    
    # We need to compute powerful_part for all numbers in [1, N+1]
    # powerful_part(n) = product of p^e for each prime p where exponent e >= 2 in factorization of n
    
    # Strategy: Sieve-based accumulation
    # For each prime p, for each power p^2, p^3, ... <= N+1:
    #   For each multiple m of p^k (where k>=2), multiply powerful_part[m] by p^(k - (k-1)) = p
    # Wait, that's not right. Let me think more carefully.
    #
    # If n = p^e * rest, and e >= 2, then the powerful part includes p^e.
    # So we need to find the full exponent e for each prime in each number.
    #
    # Better approach:
    # 1. For each prime p, find the maximum e such that p^e divides each number.
    # 2. If e >= 2, multiply powerful_part[number] by p^e.
    #
    # Sieve approach:
    # - Initialize powerful_part[1..N+1] = 1
    # - For each prime p:
    #   - For k = 2, 3, ... while p^k <= N+1:
    #     - For each multiple m of p^k (m = p^k, 2*p^k, 3*p^k, ...):
    #       - This tells us p^k divides m, but we need the FULL exponent.
    #
    # Alternative cleaner approach:
    # - For each prime p, compute the exponent of p in each number via a sieve.
    # - Then multiply p^e if e >= 2.
    #
    # Most efficient sieve approach:
    # powerful_part = array of 1s
    # For each prime p:
    #   p2 = p*p
    #   For each multiple m of p2:
    #     Count how many times p divides m (i.e., the exponent e)
    #     If e >= 2: powerful_part[m] *= p^e
    #
    # But counting exponent per multiple is expensive. Better:
    # For each prime p:
    #   For k from 2 upward while p^k <= N+1:
    #     For each multiple m of p^k but NOT p^(k+1):
    #       powerful_part[m] *= p^k
    #
    # Even better: just iterate over powers.
    # For each prime p:
    #   power = p*p
    #   while power <= N+1:
    #     for m in range(power, N+2, power):
    #       powerful_part[m] *= p  # No wait, this only adds one factor of p
    #     power *= p
    #
    # Hmm, let me think again. If p^3 divides m, then we want to multiply by p^3.
    # If we do:
    #   for k=2: multiply all multiples of p^2 by p^2
    #   for k=3: multiply all multiples of p^3 by p (additional factor)
    #   for k=4: multiply all multiples of p^4 by p (additional factor)
    # Then for a number with exponent e>=2, it gets multiplied by p^2 * p * p * ... = p^e. Correct!
    #
    # So:
    # For each prime p:
    #   pk = p*p
    #   while pk <= N+1:
    #     for m in range(pk, N+2, pk):
    #       if k == 2: powerful_part[m] *= p*p
    #       else: powerful_part[m] *= p
    #     pk *= p
    #     k += 1
    #
    # This works! Let's implement it.
    
    max_val = N + 1
    # Use float64 for accumulation, then convert to int at the end
    # Actually, let's use Python ints to avoid overflow issues, but that's slow.
    # Let's use float64 and check if values are reasonable.
    # Max powerful part: product of all p^e for p^e <= 1000001. 
    # The largest is probably around 2^19 * 3^12 * 5^8 * ... which could be huge.
    # Let's use Python's arbitrary precision via object arrays? Too slow.
    # Let's use float64 - the max product might overflow. Let's check.
    # Actually, for the purpose of finding the MAX product of consecutive powerful parts,
    # we just need to compare products. If we use log-space, we can avoid overflow.
    
    # Use log-space for safety
    log_powerful = np.zeros(max_val + 1, dtype=np.float64)
    
    # Generate primes up to max_val using simple sieve
    is_prime = np.ones(max_val + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # For each prime, accumulate the log of the powerful part contribution
    for p in primes:
        p_int = int(p)
        if p_int * p_int > max_val:
            break
        pk = p_int * p_int
        k = 2
        while pk <= max_val:
            # For k=2, multiply by p^2 -> add 2*log(p)
            # For k>2, multiply by p -> add log(p)
            if k == 2:
                add_val = 2.0 * math.log(p_int)
            else:
                add_val = math.log(p_int)
            
            # Vectorized: add to all multiples of pk
            indices = np.arange(pk, max_val + 1, pk)
            log_powerful[indices] += add_val
            
            pk *= p_int
            k += 1
        
        if time.time() - start_time > deadline - 5:
            break
    
    # Now compute the product of consecutive powerful parts in log-space
    # We want to maximize log_powerful[n] + log_powerful[n+1] for n in [LO, N]
    # which is equivalent to maximizing powerful_part[n] * powerful_part[n+1]
    
    if time.time() - start_time > deadline - 3:
        # Not enough time, return a reasonable default
        return LO
    
    # Compute sum of consecutive log-powerful values
    # valid range: n in [LO, N], so n+1 in [LO+1, N+1]
    consecutive_log = log_powerful[LO:N+1] + log_powerful[LO+1:N+2]
    
    # Find the index of the maximum
    best_idx = int(np.argmax(consecutive_log))
    best_n = LO + best_idx
    
    # Verify it's in range
    if best_n < LO or best_n > N:
        best_n = LO
    
    return best_n
# EVOLVE-BLOCK-END
