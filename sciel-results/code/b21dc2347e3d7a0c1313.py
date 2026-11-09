import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 210
    max_val = 10210

    # Sieve of Eratosthenes
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
               
    primes = [i for i, val in enumerate(is_prime) if val]
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    def get_mask(x):
        mask = 0
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count ^= 1
                if count:
                    mask |= (1 << prime_index[p])
        if temp > 1:
            if temp in prime_index:
                mask |= (1 << prime_index[temp])
        return mask

    # Precompute masks
    candidate_masks = {}
    for M in range(211, max_val + 1):
        candidate_masks[M] = get_mask(M)
        
    target_mask = get_mask(n)
    
    basis = {}
    t_vec = target_mask
    sol_mask = 0
    null_space = []

    for M in range(211, max_val + 1):
        if time.time() > deadline:
            # Fallback
            return [4*n]

        v = candidate_masks[M]
        h = 1 << (M - 211)
        
        temp_v = v
        temp_h = h
        while temp_v:
            p = temp_v.bit_length() - 1
            if p in basis:
                b_vec, b_hist = basis[p]
                temp_v ^= b_vec
                temp_h ^= b_hist
            else:
                break
        
        if temp_v == 0:
            null_space.append(temp_h)
        else:
            basis[p] = (temp_v, temp_h)
            if (t_vec >> p) & 1:
                t_vec ^= temp_v
                sol_mask ^= temp_h
        
        if t_vec == 0:
            weight = bin(sol_mask).count('1')
            if 1 <= weight <= 100:
                sol_indices = []
                hh = sol_mask
                while hh:
                    lsb = hh & -hh
                    idx = lsb.bit_length() - 1
                    sol_indices.append(211 + idx)
                    hh ^= lsb
                
                prod = n
                for x in sol_indices:
                    prod *= x
                if math.isqrt(prod)**2 == prod:
                    return [int(x) for x in sol_indices]
            
            # If weight > 100, try to reduce using null space
            # Greedy
            improved = True
            while improved:
                improved = False
                for z in null_space:
                    new_mask = sol_mask ^ z
                    if bin(new_mask).count('1') < bin(sol_mask).count('1'):
                        sol_mask = new_mask
                        improved = True
            
            weight = bin(sol_mask).count('1')
            if 1 <= weight <= 100:
                sol_indices = []
                hh = sol_mask
                while hh:
                    lsb = hh & -hh
                    idx = lsb.bit_length() - 1
                    sol_indices.append(211 + idx)
                    hh ^= lsb
                
                prod = n
                for x in sol_indices:
                    prod *= x
                if math.isqrt(prod)**2 == prod:
                    return [int(x) for x in sol_indices]

    return [4*n]
