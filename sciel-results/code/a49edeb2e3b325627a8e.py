"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    N = 1000000
    LOW = 10000
    
    # powerful_part[i] will store the powerful part of i
    # Initialize to 1 (empty product)
    powerful_part = np.ones(N + 1, dtype=np.int64)
    
    # Sieve: for each prime p, for each power p^e with e>=2,
    # multiply powerful_part[multiple] by p^e if p^e divides the multiple
    # But we need the FULL p^e where e is the actual exponent in the factorization
    
    # Better approach: for each number, we need to find all prime factors with exponent >= 2
    # and take the full prime power (p^e where e is the total exponent)
    
    # Use a modified sieve:
    # For each prime p, for each multiple m of p, compute the exponent of p in m
    # If exponent >= 2, multiply powerful_part[m] by p^exponent
    
    # Efficient way: for each prime p, iterate through multiples
    # For a multiple m = k*p, the exponent of p in m is 1 + exponent of p in k
    # This is tricky. Let's use a different approach.
    
    # For each prime p, consider p^2, p^3, ...
    # For p^2: all multiples of p^2 have at least p^2
    # For p^3: all multiples of p^3 have at least p^3
    # We want the HIGHEST power p^e that divides m, where e >= 2
    
    # Approach: for each prime p, for e = 2, 3, ... while p^e <= N:
    #   For each multiple m of p^e that is NOT a multiple of p^(e+1):
    #     The exponent of p in m is exactly e
    #     So multiply powerful_part[m] by p^e
    
    # But this is complex. Alternative: 
    # For each prime p, for each multiple m of p:
    #   Count how many times p divides m (exponent e)
    #   If e >= 2, multiply powerful_part[m] by p^e
    
    # Let's use a factorization-based sieve
    # smallest_prime_factor array
    spf = np.zeros(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= N:
                for j in range(i * i, N + 1, i):
                    if spf[j] == 0:
                        spf[j] = i
    
    # Now compute powerful part for each number
    # For each number m, factorize using spf and collect prime powers with e >= 2
    # This might be slow for all numbers. Let's think of a better way.
    
    # Alternative: precompute for each prime p, the list of numbers where p^e (e>=2) divides them
    # and the exact exponent.
    
    # Let's do it prime by prime:
    # For each prime p:
    #   p2 = p*p
    #   For e = 2, 3, ...:
    #     pe = p^e
    #     if pe > N: break
    #     pe_next = pe * p
    #     For each multiple m of pe that is NOT a multiple of pe_next:
    #       # This means the exponent of p in m is exactly e
    #       powerful_part[m] *= pe
    #     # Also, multiples of pe_next will be handled at higher e
    
    # Wait, this isn't quite right. Let me reconsider.
    # For a number m, the exponent of p in m is the largest e such that p^e | m.
    # So for each prime p, I want to find for each m the exponent e_p(m).
    # If e_p(m) >= 2, then p^{e_p(m)} contributes to powerful_part[m].
    
    # Efficient sieve approach:
    # For each prime p, let's process multiples of p.
    # For m = p * k, the exponent of p in m is 1 + exponent of p in k.
    # This recursive structure is hard to use directly.
    
    # Let's just do it with the spf factorization for each number, but optimize.
    # Actually, let's use a different sieve:
    
    # powerful_part initialized to 1
    # For each prime p:
    #   For each power p^e with e >= 2 and p^e <= N:
    #     For each multiple m of p^e:
    #       Check if p^(e+1) does NOT divide m (i.e., m / p^e is not divisible by p)
    #       If so, the exponent of p in m is exactly e, so multiply powerful_part[m] by p^e
    #       If p^(e+1) does divide m, then the exponent is higher, handled later
    
    # This is still complex. Let me just factorize each number using spf.
    
    # Given time constraints, let's factorize each number in [LOW, N]
    # and compute its powerful part.
    
    # But that's 990000 numbers, each factorization is O(log n) with spf.
    # Should be fast enough.
    
    best_n = LOW
    best_product = 0
    
    # Compute powerful part for each number from LOW to N
    # We need powerful_part for n and n+1, so compute from LOW to N
    pp = np.ones(N + 1, dtype=np.int64)
    
    for m in range(2, N + 1):
        if spf[m] == 0:
            spf[m] = m
    
    # Recompute spf properly
    spf = np.zeros(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= N:
                for j in range(i * i, N + 1, i):
                    if spf[j] == 0:
                        spf[j] = i
    
    # Now factorize each number and compute powerful part
    for m in range(2, N + 1):
        x = m
        pp_val = 1
        while x > 1:
            p = spf[x]
            # Count exponent of p in x
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                pp_val *= (p ** e)
        pp[m] = pp_val
    
    # Now find max of pp[n] * pp[n+1] for n in [LOW, N-1]
    products = pp[LOW:N] * pp[LOW+1:N+1]
    idx = np.argmax(products)
    best_n = LOW + idx
    best_product = products[idx]
    
    return int(best_n)
# EVOLVE-BLOCK-END
