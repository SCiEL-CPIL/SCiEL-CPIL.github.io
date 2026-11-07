import math

def construct():
    N_MAX = 10210
    N_MIN = 211
    
    # 1. Sieve of Eratosthenes
    is_prime = [True] * (N_MAX + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N_MAX)) + 1):
        if is_prime[i]:
            for j in range(i*i, N_MAX + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # 2. Compute bitsets for numbers in range
    # bitset[x] will be an integer where bit i is 1 if prime i divides x with odd exponent
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
            # temp is a prime
            b |= (1 << prime_to_idx[temp])
        num_bitsets[x] = b
        
    # 3. Target bitset for 210
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
        
    # 4. Iterate M and maintain basis
    basis = [] # list of (vec, mask)
    
    # mask will have bit i set if number (N_MIN + i) is used
    for M in range(N_MIN, N_MAX + 1):
        v = num_bitsets[M]
        m = 1 << (M - N_MIN)
        
        current_v = v
        current_m = m
        
        # Reduce current_v by existing basis
        # Basis is sorted by bit_length descending
        for b_v, b_m in basis:
            # Find highest set bit in b_v
            # Since basis is in "reduced" form relative to each other?
            # Actually, the standard algorithm:
            # We check if current_v has the same highest bit as b_v?
            # No, we check if current_v has the pivot bit of b_v.
            # To make this fast, we can just check if current_v & b_v has the highest bit of b_v set?
            # Simpler: Just iterate. If current_v has the bit set that is the highest bit of b_v, reduce.
            # But we need to know the highest bit of b_v.
            pivot_bit = 1 << (b_v.bit_length() - 1)
            if current_v & pivot_bit:
                current_v ^= b_v
                current_m ^= b_m
                
        if current_v != 0:
            basis.append((current_v, current_m))
            # Sort to keep highest bits first
            basis.sort(key=lambda x: x[0].bit_length(), reverse=True)
            
        # Check if target is in span
        w = target
        result_mask = 0
        for b_v, b_m in basis:
            pivot_bit = 1 << (b_v.bit_length() - 1)
            if w & pivot_bit:
                w ^= b_v
                result_mask ^= b_m
                
        if w == 0:
            # Found solution
            # Extract indices from result_mask
            selected = []
            for i in range(M - N_MIN + 1):
                if (result_mask >> i) & 1:
                    selected.append(N_MIN + i)
            return selected
            
    # Fallback (should not happen)
    return [840]
