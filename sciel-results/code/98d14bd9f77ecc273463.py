"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin
    
    N = 1000000
    low = 10000
    
    # Sieve of Eratosthenes to find primes
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # P[n] = powerful part of n = product of p^e for primes p where v_p(n) >= 2
    P = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        pe = p2  # p^e, starting from p^2
        e = 2
        while pe <= N:
            pe_next = pe * p  # p^(e+1)
            # For multiples of pe that are NOT multiples of pe_next:
            # v_p(m) = e exactly
            # Step through multiples of pe, skip those divisible by pe_next
            # Multiples of pe: pe, 2*pe, 3*pe, ...
            # Among these, those divisible by pe_next = pe*p are: pe*p, 2*pe*p, ...
            # i.e., every p-th multiple of pe
            # So we want multiples of pe where the multiplier is NOT divisible by p
            # multiplier k: k*pe, we want k not divisible by p
            # k = 1, 2, ..., floor(N/pe), skip k divisible by p
            max_k = N // pe
            # Generate all k from 1 to max_k, skip multiples of p
            # For speed, process in blocks
            k = 1
            while k <= max_k:
                m = k * pe
                if m <= N:
                    P[m] *= pe
                k += 1
                # Skip if next k would be divisible by p
                if k % p == 0:
                    k += 1
            # Next power
            if pe_next > N:
                break
            pe = pe_next
            e += 1
    
    # Find the best n in [low, N-1] such that P[n]*P[n+1] is maximized
    # We need n and n+1 both in range, so n in [low, N-1]
    products = P[low:N] * P[low+1:N+1]
    
    # Find the index of the maximum
    best_idx = np.argmax(products)
    best_n = low + best_idx
    best_val = products[best_idx]
    
    elapsed = time.time() - start_time
    if elapsed > 70:
        # Safety: return best found
        pass
    
    return int(best_n)
# EVOLVE-BLOCK-END
