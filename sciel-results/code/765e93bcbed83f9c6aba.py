# EVOLVE-BLOCK-START
import math

def construct():
    N = 210
    MAX_VAL = 10210
    MAX_SIZE = 100
    
    # 1. Sieve of Eratosthenes for primes up to MAX_VAL
    limit = MAX_VAL
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, p in enumerate(is_prime) if p]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # 2. Precompute square-free kernel bitmask for each number 2..MAX_VAL
    # kernel[x] is an integer bitmask where bit i is set if prime[i] appears to an odd power in x
    kernel = [0] * (limit + 1)
    for x in range(2, limit + 1):
        mask = 0
        temp = x
        for p in primes:
            if p * p > temp:
                break
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count % 2 == 1:
                mask |= (1 << prime_to_idx[p])
        if temp > 1:
            mask |= (1 << prime_to_idx[temp])
        kernel[x] = mask
        
    # Target mask for 210
    target_mask = kernel[210]
    
    # 3. Gaussian Elimination Basis
    # basis[hb] = vector with highest bit hb
    # combo[hb] = bitmask of indices (relative to 211) that form this basis vector
    basis = [None] * num_primes
    combo = [0] * num_primes
    
    def add_to_basis(vec, idx_mask):
        v = vec
        c = idx_mask
        while v:
            hb = v.bit_length() - 1
            if basis[hb] is None:
                basis[hb] = v
                combo[hb] = c
                return True
            else:
                v ^= basis[hb]
                c ^= combo[hb]
        return False
        
    def is_in_span(target):
        v = target
        c = 0
        while v:
            hb = v.bit_length() - 1
            if basis[hb] is None:
                return False, 0
            v ^= basis[hb]
            c ^= combo[hb]
        return True, c

    # 4. Iterate M from 211 to MAX_VAL
    for M in range(211, limit + 1):
        idx_mask = 1 << (M - 211)
        add_to_basis(kernel[M], idx_mask)
        
        found, sol_mask = is_in_span(target_mask)
        if found:
            selected = []
            m = sol_mask
            i = 0
            while m:
                if m & 1:
                    selected.append(211 + i)
                m >>= 1
                i += 1
            
            # Verify size constraint
            if len(selected) <= MAX_SIZE:
                # Verify perfect square condition
                prod = 1
                for x in selected:
                    prod *= x
                total = prod * N
                root = math.isqrt(total)
                if root * root == total:
                    return [int(x) for x in selected]
                else:
                    # Logic error in kernel or span?
                    # Continue searching?
                    # For safety, if verification fails, we should not return.
                    # But this should not happen.
                    pass
            else:
                # If size > 100, we cannot use this solution.
                # We must continue to next M.
                # Note: For M < 311, size is always <= 100.
                pass

    # Fallback
    return [840]
# EVOLVE-BLOCK-END
