"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import numpy as np

def construct():
    N_MIN = 10000
    N_MAX = 1000000
    LIMIT = N_MAX + 1
    
    # 1. Sieve for Smallest Prime Factor (SPF) up to LIMIT
    # This allows O(log n) factorization for any n <= LIMIT
    spf = np.zeros(LIMIT + 1, dtype=np.int64)
    for i in range(2, LIMIT + 1):
        if spf[i] == 0:
            spf[i] = i
            for j in range(i * i, LIMIT + 1, i):
                if spf[j] == 0:
                    spf[j] = i
    
    # Helper function to compute Powerful Part of n using SPF
    def powerful_part(n):
        if n < 2:
            return 1
        pp = 1
        while n > 1:
            p = spf[n]
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            if count >= 2:
                pp *= (p ** count)
        return pp

    # 2. Generate all powerful numbers up to LIMIT
    # A powerful number is of the form a^2 * b^3
    # We can generate them by iterating over squares and cubes
    powerful_numbers = set()
    # Max square root is sqrt(LIMIT) ~ 1000
    # Max cube root is cbrt(LIMIT) ~ 100
    # Actually, we can just iterate i from 1 to sqrt(LIMIT) and j from 1 to cbrt(LIMIT)
    # n = i^2 * j^3
    max_i = int(math.isqrt(LIMIT))
    max_j = int(round(LIMIT ** (1/3))) + 1
    
    for j in range(1, max_j + 1):
        j3 = j ** 3
        if j3 > LIMIT:
            break
        for i in range(1, max_i + 1):
            n = i * i * j3
            if n > LIMIT:
                break
            powerful_numbers.add(n)
            
    powerful_list = sorted(powerful_numbers)
    
    # 3. Iterate over pairs (A, B) of powerful numbers
    best_n = N_MIN
    max_product = 0
    
    # Optimization: Only consider A, B that are "large" enough?
    # Or just iterate all. len(powerful_list) is approx 1000.
    # 1000 * 1000 = 1,000,000 pairs.
    # For each pair, we do GCD, CRT, and 2 powerful_part calls.
    # powerful_part is fast (log n steps).
    # This should be well within 60s.
    
    for i in range(len(powerful_list)):
        A = powerful_list[i]
        for j in range(len(powerful_list)):
            B = powerful_list[j]
            
            # We want A * B to be large, so we can skip small ones?
            # But we need to check all to be sure.
            # However, if A*B is small, it's unlikely to beat the current best.
            # Let's just run all.
            
            if math.gcd(A, B) != 1:
                continue
                
            # Solve n = k * A such that n = -1 (mod B)
            # k * A = m * B - 1
            # k * A + 1 = m * B
            # We need A * k = B - 1 (mod B)
            # k = (B - 1) * A^-1 (mod B)
            
            try:
                A_inv = pow(A, -1, B)
            except ValueError:
                # This should not happen if gcd(A,B)=1, but just in case
                continue
                
            k_0 = ((B - 1) * A_inv) % B
            if k_0 == 0:
                k_0 = B # k must be positive? n = k*A. If k=0, n=0.
                # Actually k_0 is in [0, B-1]. If k_0=0, n=0.
                # Next solution is k = B.
            
            n_0 = k_0 * A
            mod = A * B
            
            # Find smallest n >= N_MIN
            # n = n_0 + t * mod
            if n_0 < N_MIN:
                t = math.ceil((N_MIN - n_0) / mod)
                n = n_0 + t * mod
            else:
                n = n_0
                
            if n > N_MAX:
                continue
                
            # Calculate actual powerful parts
            pp_n = powerful_part(n)
            pp_np1 = powerful_part(n + 1)
            
            prod = pp_n * pp_np1
            if prod > max_product:
                max_product = prod
                best_n = n
                
    return best_n
# EVOLVE-BLOCK-END
