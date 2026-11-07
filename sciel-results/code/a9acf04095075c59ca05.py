import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N_MIN = 211
    limit = 10210
    
    # Precompute primes up to limit
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, p in enumerate(is_prime) if p]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Function to get square-free part bitset of x
    def get_sf_bitset(x):
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
        return b
    
    target = get_sf_bitset(210)
    
    # Basis indexed by pivot (highest set bit)
    # basis_vec[pivot] = (vec, mask)
    basis_vec = [None] * num_primes
    
    # We iterate M from N_MIN upwards
    for M in range(N_MIN, limit + 1):
        if time.time() > deadline:
            break
            
        v = get_sf_bitset(M)
        idx = M - N_MIN
        m = 1 << idx
        
        # Reduce v using existing basis
        current_v = v
        current_m = m
        
        # Reduce by iterating bits from high to low
        while current_v:
            pivot = current_v.bit_length() - 1
            if basis_vec[pivot] is not None:
                b_vec, b_mask = basis_vec[pivot]
                current_v ^= b_vec
                current_m ^= b_mask
            else:
                break
        
        if current_v != 0:
            # Find pivot of current_v
            pivot = current_v.bit_length() - 1
            # Insert into basis
            basis_vec[pivot] = (current_v, current_m)
            
        # Check if target is in span
        w = target
        res_mask = 0
        while w:
            pivot = w.bit_length() - 1
            if basis_vec[pivot] is not None:
                b_vec, b_mask = basis_vec[pivot]
                w ^= b_vec
                res_mask ^= b_mask
            else:
                break
        
        if w == 0:
            # Found solution
            selected = []
            r = res_mask
            i = 0
            while r:
                if r & 1:
                    selected.append(N_MIN + i)
                r >>= 1
                i += 1
            
            # Verify
            prod = 210
            for x in selected:
                prod *= x
            s = math.isqrt(prod)
            if s * s == prod:
                # Ensure types are correct
                return list(selected)
            else:
                pass
                
    # Fallback if not found (should not happen)
    return [840]
