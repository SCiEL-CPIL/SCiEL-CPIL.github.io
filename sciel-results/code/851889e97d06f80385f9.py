import math
import time

def construct():
    t0 = time.time()
    DEADLINE = t0 + 75.0
    
    N_MAX = 10210
    N_MIN = 211
    
    # Sieve of Eratosthenes
    is_prime = [True] * (N_MAX + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N_MAX)) + 1):
        if is_prime[i]:
            for j in range(i*i, N_MAX + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    n_primes = len(primes)
    
    # Precompute bitsets for all numbers in range
    num_bitsets = {}
    for x in range(N_MIN, N_MAX + 1):
        temp = x
        b = 0
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    b |= (1 << prime_to_idx[p])
        if temp > 1:
            b |= (1 << prime_to_idx[temp])
        num_bitsets[x] = b
        
    # Target bitset for 210
    target = 0
    temp = 210
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count % 2 == 1:
                target |= (1 << prime_to_idx[p])
    if temp > 1:
        target |= (1 << prime_to_idx[temp])
        
    # Basis for GF(2) elimination
    # basis_list[i] = (vector, mask) where vector has highest set bit at position i
    basis_list = [None] * n_primes
    
    # Track the reduction of target incrementally
    # target_reduced: the current reduced form of target
    # target_mask: the mask that would give us the solution
    target_reduced = target
    target_mask = 0
    
    def reduce_with_basis(v, m, basis):
        """Reduce vector v using basis, tracking mask m"""
        while v != 0:
            pivot_pos = v.bit_length() - 1
            if pivot_pos < len(basis) and basis[pivot_pos] is not None:
                bv, bm = basis[pivot_pos]
                v ^= bv
                m ^= bm
            else:
                break
        return v, m
    
    best_result = None
    
    for M in range(N_MIN, N_MAX + 1):
        v = num_bitsets[M]
        m = 1 << (M - N_MIN)
        
        # Reduce v using current basis
        v, m = reduce_with_basis(v, m, basis_list)
        
        if v != 0:
            pivot_pos = v.bit_length() - 1
            basis_list[pivot_pos] = (v, m)
            
            # If this new basis vector could help reduce target, update target_reduced
            # Check if target_reduced has this pivot bit set
            if (target_reduced >> pivot_pos) & 1:
                target_reduced ^= v
                target_mask ^= m
        
        # Check if target is now in span (target_reduced == 0)
        if target_reduced == 0:
            # Found a solution
            selected = []
            for i in range(M - N_MIN + 1):
                if (target_mask >> i) & 1:
                    selected.append(N_MIN + i)
            if len(selected) >= 1:
                return [int(x) for x in selected]
        
        # Time check
        if time.time() > DEADLINE:
            break
    
    # Fallback
    return [840]
