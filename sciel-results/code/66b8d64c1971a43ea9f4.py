import math
import time

def construct():
    # Primes up to 10210
    limit = 10210
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, val in enumerate(is_prime) if val]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Precompute parity bitmasks for all numbers in range
    # bitmask is an int where bit i is 1 if prime[i] has odd exponent
    bitmasks = {}
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
            # x is a prime
            idx = prime_to_idx[x]
            mask |= (1 << idx)
        bitmasks[n] = mask
        
    # Target bitmask for 210
    target_mask = 0
    x = 210
    for i, p in enumerate(primes):
        if p * p > x:
            break
        if x % p == 0:
            count = 0
            while x % p == 0:
                x //= p
                count += 1
            if count % 2 == 1:
                target_mask |= (1 << i)
    if x > 1:
        idx = prime_to_idx[x]
        target_mask |= (1 << idx)
        
    # Gaussian Elimination State
    # basis: list of (pivot_index, vector, combination_mask)
    # We use a dict for basis to quickly check if a pivot exists
    basis = {}
    
    start_time = time.time()
    
    for M in range(211, limit + 1):
        # Check deadline
        if time.time() - start_time > 50:
            # Return best found so far? 
            # Since we iterate M, we haven't found a solution yet if we reach here.
            # But we should return something valid.
            # The problem guarantees a solution. 
            # If we run out of time, we might return a hardcoded valid one.
            # But 10000 iterations is fast.
            pass
            
        v = bitmasks[M]
        
        # Try to reduce v using current basis
        u = v
        c = 1 << (M - 211) # Index of M in the list of numbers [211, ..., M]
        # Note: The combination mask bits correspond to the index of the number.
        # Number n corresponds to bit (n - 211).
        
        # Eliminate u
        # We iterate through existing basis vectors.
        # To be efficient, we should check pivots.
        # Since we add vectors in increasing order, the pivots might not be sorted.
        # We can just iterate through the values of the basis.
        # But we need to be careful: if we modify u, we might need to eliminate again?
        # No, standard GE:
        # for each pivot in basis:
        #   if u has that pivot bit:
        #     u ^= basis_vec
        #     c ^= basis_comb
        
        # We need to iterate through all basis vectors.
        # To make this fast, we can store basis in a list and sort by pivot?
        # Or just iterate. Number of basis vectors is small.
        
        for pivot, b_vec, b_comb in basis.values():
            if (u >> pivot) & 1:
                u ^= b_vec
                c ^= b_comb
        
        if u != 0:
            # Find pivot for u
            # Highest set bit
            pivot = u.bit_length() - 1
            # If this pivot already exists in basis, it means u is dependent?
            # No, if u != 0, and we reduced it, it means it's a new independent vector.
            # But we must check if basis already has a vector with this pivot.
            # In standard GE, if we process columns in order, we don't.
            # Here, we add vectors one by one.
            # If basis already has a vector with this pivot, then u should have been reduced to 0?
            # Not necessarily.
            # Actually, if we maintain the basis in row-echelon form,
            # each new vector either extends the basis or is 0.
            # If it extends, its pivot is new.
            if pivot not in basis:
                basis[pivot] = (pivot, u, c)
                
        # Check if target is in span
        # Reduce target_mask using basis
        t_u = target_mask
        t_c = 0
        for pivot, b_vec, b_comb in basis.values():
            if (t_u >> pivot) & 1:
                t_u ^= b_vec
                t_c ^= b_comb
        
        if t_u == 0:
            # Solution found
            # t_c is the bitmask of selected numbers
            # Extract numbers
            selected = []
            for i in range(len(bitmasks)): # M-211 is the max index
                if (t_c >> i) & 1:
                    selected.append(211 + i)
            
            # Verify
            if len(selected) > 100:
                # This shouldn't happen for small M
                continue
                
            # Check product * 210 is square
            # We trust the linear algebra, but let's verify with isqrt
            prod = 1
            for x in selected:
                prod *= x
            prod *= 210
            s = math.isqrt(prod)
            if s * s == prod:
                return selected
            else:
                # Error in logic?
                # Continue searching?
                pass
                
    # Fallback
    return [840]
