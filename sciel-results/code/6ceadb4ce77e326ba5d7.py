import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin before 90s kill
    
    limit = 10210
    
    # Sieve primes up to limit
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, p in enumerate(is_prime) if p]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    def get_mask(n):
        mask = 0
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    mask ^= (1 << prime_to_idx[p])
        if temp > 1:
            mask ^= (1 << prime_to_idx[temp])
        return mask
    
    target_mask = get_mask(210)
    
    # RREF basis: basis[pivot] = (vector_mask, combo_mask)
    # We use a dict for O(1) pivot lookup
    basis = {}  # pivot_bit -> (vector_mask, combo_mask)
    
    def reduce_and_insert(v, c):
        """
        Reduce v against existing basis (descending pivots).
        If result is nonzero, insert into basis and update existing entries.
        Returns (reduced_v, reduced_c).
        """
        while v != 0:
            pivot = v.bit_length() - 1
            if pivot in basis:
                b_mask, b_combo = basis[pivot]
                v ^= b_mask
                c ^= b_combo
            else:
                # Insert new basis vector
                basis[pivot] = (v, c)
                # Update all existing basis vectors to eliminate this pivot
                for p2, (vm2, cm2) in list(basis.items()):
                    if p2 != pivot and (vm2 >> pivot) & 1:
                        basis[p2] = (vm2 ^ v, cm2 ^ c)
                break
        return v, c
    
    def check_in_span(target):
        """
        Check if target is in span of current basis.
        Returns (in_span, combo_mask).
        """
        curr_v = target
        curr_c = 0
        while curr_v != 0:
            pivot = curr_v.bit_length() - 1
            if pivot in basis:
                b_mask, b_combo = basis[pivot]
                curr_v ^= b_mask
                curr_c ^= b_combo
            else:
                return False, 0
        return True, curr_c
    
    for M in range(211, limit + 1):
        if time.time() > deadline:
            break
        
        idx = M - 211
        v = get_mask(M)
        c = 1 << idx
        
        reduce_and_insert(v, c)
        
        in_span, combo = check_in_span(target_mask)
        
        if in_span:
            # Reconstruct solution
            sol_indices = []
            for i in range(idx + 1):
                if combo & (1 << i):
                    sol_indices.append(i + 211)
            
            if 1 <= len(sol_indices) <= 100:
                # Verify solution
                prod = 210
                for x in sol_indices:
                    prod *= x
                root = math.isqrt(prod)
                if root * root == prod:
                    return [int(x) for x in sol_indices]
    
    # Fallback
    return [224, 240]
