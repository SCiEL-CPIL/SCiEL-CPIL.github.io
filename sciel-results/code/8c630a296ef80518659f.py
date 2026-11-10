"""Improved program using Squarefree Sieve Filtering."""
# EVOLVE-BLOCK-START
import math

def construct():
    MAX_N = 1000000
    MIN_N = 10000
    
    # 1. Build Smallest Prime Factor (SPF) table
    spf = list(range(MAX_N + 2))
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_N + 2, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Identify non-squarefree numbers
    # A number is squarefree if it is not divisible by p^2 for any prime p.
    is_sqfree = [True] * (MAX_N + 2)
    is_sqfree[0] = False
    is_sqfree[1] = True # 1 is squarefree
    
    for p in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[p] == p: # p is prime
            p2 = p * p
            for j in range(p2, MAX_N + 2, p2):
                is_sqfree[j] = False
    
    # 3. Function to compute powerful part
    def powerful_part(n):
        pp = 1
        while n > 1:
            p = spf[n]
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            if e >= 2:
                pp *= (p ** e)
        return pp
    
    best_n = MIN_N
    best_product = 0
    
    # 4. Scan only candidates where n or n+1 is not squarefree
    # If both are squarefree, product is 1.
    for n in range(MIN_N, MAX_N):
        if is_sqfree[n] and is_sqfree[n+1]:
            continue
        
        # Compute powerful parts
        # Optimization: Cache results? 
        # Since we skip many, simple calculation is fine.
        pp_n = powerful_part(n)
        pp_n1 = powerful_part(n + 1)
        
        prod = pp_n * pp_n1
        if prod > best_product:
            best_product = prod
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
