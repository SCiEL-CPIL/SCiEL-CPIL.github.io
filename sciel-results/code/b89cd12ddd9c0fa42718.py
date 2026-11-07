import math
import time

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 85.0  # 85s deadline for 90s wall-clock limit
    
    # Primes up to 10210
    max_val = 10210
    # Sieve of Eratosthenes
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    prime_index = {p: i for i, p in enumerate(primes)}
    
    # Calculate masks for all numbers
    # mask[n] is the bitmask of prime indices in the square-free part of n
    masks = [0] * (max_val + 1)
    for n in range(2, max_val + 1):
        temp = n
        mask = 0
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    mask ^= (1 << prime_index[p])
        if temp > 1:
            # temp is a prime
            mask ^= (1 << prime_index[temp])
        masks[n] = mask
        
    target_mask = masks[210]
    
    # Basis storage
    # Each entry: (pivot_bit, vector_mask, coeff_mask)
    # pivot_bit is the index of the highest set bit in vector_mask
    # coeff_mask is a bitmask of the original numbers (211, 212, ...) used
    # We map the original number x to an index idx = x - 211
    # So coeff_mask bit idx is 1 if x is in the combination
    
    basis = {} # pivot_bit -> (vector_mask, coeff_mask)
    
    # We iterate M from 211 upwards
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break
            
        v = masks[M]
        idx = M - 211
        c = 1 << idx
        
        # Reduce v by existing basis
        temp_v = v
        temp_c = c
        while temp_v:
            pivot = temp_v.bit_length() - 1
            if pivot in basis:
                b_vec, b_coeff = basis[pivot]
                temp_v ^= b_vec
                temp_c ^= b_coeff
            else:
                # Add to basis
                basis[pivot] = (temp_v, temp_c)
                break
        
        # Check if target is in span
        check_v = target_mask
        check_c = 0
        while check_v:
            pivot = check_v.bit_length() - 1
            if pivot in basis:
                b_vec, b_coeff = basis[pivot]
                check_v ^= b_vec
                check_c ^= b_coeff
            else:
                break
        
        if check_v == 0:
            # Solution found
            num_selected = check_c.bit_count()
            if num_selected <= 100:
                result = []
                c_mask = check_c
                while c_mask:
                    lsb = c_mask & -c_mask
                    idx_found = lsb.bit_length() - 1
                    x_found = idx_found + 211
                    result.append(x_found)
                    c_mask ^= lsb
                return result
            else:
                pass
                
    # Fallback: return a trivial solution if loop finishes (should not happen)
    # 210 * (4*210) = 210^2 * 4 = (2*210)^2. 
    # 4*210 = 840.
    # 840 is <= 10210.
    # Distinct? Yes, just one number.
    # Strictly larger than 210? Yes.
    return [840]

# EVOLVE-BLOCK-END
