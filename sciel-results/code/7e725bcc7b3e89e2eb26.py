import math
import time

def construct():
    limit = 10210
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Sieve for primes up to limit
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute parity bitmasks for all numbers 211..limit
    # Use a compact representation: for each n, store the bitmask as an int
    bitmasks = [0] * (limit + 1)
    for n in range(211, limit + 1):
        mask = 0
        x = n
        for i, p in enumerate(primes):
            if p * p > x:
                break
            if x % p == 0:
                count = 0
                while x % p == 0:
                    x //= p
                    count += 1
                if count % 2 == 1:
                    mask |= (1 << i)
        if x > 1:
            idx = prime_to_idx[x]
            mask |= (1 << idx)
        bitmasks[n] = mask
    
    # Target: parity of 210 = 2^1 * 3^1 * 5^1 * 7^1
    # So target_mask has bits set for primes 2,3,5,7
    target_mask = 0
    for p in [2, 3, 5, 7]:
        target_mask |= (1 << prime_to_idx[p])
    
    # Gaussian elimination over F2
    # basis: list of (pivot_bit, vector, combination)
    # We'll maintain it in reduced row echelon form
    # basis_by_pivot: dict pivot -> (vec, comb)
    basis_by_pivot = {}
    # Also track the list of vectors for quick iteration
    basis_vecs = []  # list of (pivot, vec, comb)
    
    def reduce_vec(v, comb):
        """Reduce v using basis, return (reduced_v, reduced_comb)"""
        for pivot, bvec, bcomb in basis_vecs:
            if (v >> pivot) & 1:
                v ^= bvec
                comb ^= bcomb
        return v, comb
    
    def add_to_basis(vec, comb):
        """Add vector to basis, maintaining RREF. Return True if added."""
        rvec, rcomb = reduce_vec(vec, comb)
        if rvec == 0:
            return False
        pivot = rvec.bit_length() - 1
        # Eliminate this pivot from existing basis vectors
        new_basis_vecs = []
        for p, bv, bc in basis_vecs:
            if (bv >> pivot) & 1:
                bv ^= rvec
                bc ^= rcomb
            new_basis_vecs.append((p, bv, bc))
        new_basis_vecs.append((pivot, rvec, rcomb))
        # Sort by pivot descending for efficient reduction
        new_basis_vecs.sort(key=lambda x: x[0], reverse=True)
        return new_basis_vecs
    
    for M in range(211, limit + 1):
        if time.time() > deadline:
            # Fallback: return a known valid solution
            return [840]
        
        v = bitmasks[M]
        c = 1 << (M - 211)
        
        # Reduce v
        rv, rc = reduce_vec(v, c)
        
        if rv != 0:
            # New pivot: add to basis
            new_basis = []
            for p, bv, bc in basis_vecs:
                if (bv >> rv.bit_length()) & 1:
                    # This shouldn't happen since we reduced, but handle anyway
                    pass
                # Eliminate new pivot from existing
                new_pivot = rv.bit_length() - 1
                if (bv >> new_pivot) & 1:
                    bv ^= rv
                    bc ^= rc
                new_basis.append((p, bv, bc))
            new_basis.append((rv.bit_length() - 1, rv, rc))
            new_basis.sort(key=lambda x: x[0], reverse=True)
            basis_vecs = new_basis
        
        # Check if target is in span
        tv, tc = reduce_vec(target_mask, 0)
        
        if tv == 0:
            # Solution found! Extract selected numbers
            selected = []
            for i in range(M - 211 + 1):
                if (tc >> i) & 1:
                    selected.append(211 + i)
            
            if 1 <= len(selected) <= 100:
                # Verify
                prod = 210
                for x in selected:
                    prod *= x
                s = math.isqrt(prod)
                if s * s == prod:
                    return list(selected)
            else:
                # Too many elements, continue searching
                continue
    
    # Fallback
    return [840]
