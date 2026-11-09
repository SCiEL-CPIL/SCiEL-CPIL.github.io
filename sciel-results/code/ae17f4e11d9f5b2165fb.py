import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 210
    max_val = 10210
    max_size = 100
    
    # Sieve of Eratosthenes
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    prime_index = {p: i for i, p in enumerate(primes)}

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

    target_mask = get_mask(n)
    basis = {}
    null_space_vectors = []

    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break
            
        v = get_mask(M)
        h = 1 << (M - 211)
        
        tv = v
        used_pivots = []
        while tv:
            p = tv.bit_length() - 1
            if p in basis:
                used_pivots.append(p)
                tv ^= basis[p][0]
            else:
                basis[p] = (tv, h)
                break
        
        if tv == 0:
            comb_mask = h
            for p in used_pivots:
                comb_mask ^= basis[p][1]
            null_space_vectors.append(comb_mask)
        
        # Check if target is in span
        t_target = target_mask
        t_target_hist = 0
        while t_target:
            p = t_target.bit_length() - 1
            if p in basis:
                t_target ^= basis[p][0]
                t_target_hist ^= basis[p][1]
            else:
                break
        
        if t_target == 0:
            current_sol = t_target_hist
            current_weight = current_sol.bit_count()
            
            if current_weight <= max_size:
                sol_indices = []
                hh = current_sol
                while hh:
                    lsb = hh & -hh
                    idx = lsb.bit_length() - 1
                    sol_indices.append(211 + idx)
                    hh ^= lsb
                return [int(x) for x in sol_indices]
            
            # Try to reduce weight
            # Use randomization to escape local minima
            for _ in range(50):
                changed = False
                # Shuffle null space vectors for randomization
                # We only need to consider a subset if list is large
                if len(null_space_vectors) > 50:
                    nvs = random.sample(null_space_vectors, 50)
                else:
                    nvs = null_space_vectors
                
                random.shuffle(nvs)
                
                for nv in nvs:
                    new_sol = current_sol ^ nv
                    new_weight = new_sol.bit_count()
                    if new_weight < current_weight:
                        current_sol = new_sol
                        current_weight = new_weight
                        changed = True
                
                if not changed:
                    break
                if current_weight <= max_size:
                    break
            
            if current_weight <= max_size:
                sol_indices = []
                hh = current_sol
                while hh:
                    lsb = hh & -hh
                    idx = lsb.bit_length() - 1
                    sol_indices.append(211 + idx)
                    hh ^= lsb
                return [int(x) for x in sol_indices]

    return [4*n]
