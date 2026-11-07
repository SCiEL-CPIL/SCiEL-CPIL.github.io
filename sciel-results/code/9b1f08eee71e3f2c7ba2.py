"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import math

def construct():
    n = 40
    candidates = list(range(2, n + 1))
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Precompute prime factorization vectors for each candidate
    vectors = {}
    for x in candidates:
        vec = [0] * num_primes
        temp = x
        for i, p in enumerate(primes):
            while temp % p == 0:
                vec[i] += 1
                temp //= p
        vectors[x] = tuple(vec)
    
    zero_vec = tuple([0] * num_primes)
    sums = {zero_vec}
    selected = []
    
    for a in candidates:
        v = vectors[a]
        # Check if adding v causes collision
        new_sums = set(sums)
        ok = True
        for s in sums:
            # Add v to s
            new_s = tuple(s[i] + v[i] for i in range(num_primes))
            if new_s in new_sums:
                ok = False
                break
            new_sums.add(new_s)
        
        if ok:
            selected.append(a)
            sums = new_sums
            
    return selected

# EVOLVE-BLOCK-END
