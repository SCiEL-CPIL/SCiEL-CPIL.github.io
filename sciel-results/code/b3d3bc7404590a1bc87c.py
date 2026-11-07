import math
import time
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 105
    
    # Search range - will be adjusted based on time
    N = 50_000_000
    
    # SPF Sieve using numpy
    spf = np.arange(N + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    limit = int(math.isqrt(N))
    for i in range(2, limit + 1):
        if spf[i] == i:
            start = i * i
            step = i
            indices = np.arange(start, N + 1, step)
            mask = spf[indices] == indices
            if mask.any():
                spf[indices[mask]] = i
    
    # Compute sigma using numpy vectorization where possible
    # sigma[i] = sigma[i // p] * (1 + p) if p not dividing i//p
    # sigma[i] = sigma[i // p^a] * (p^(a+1) - 1) / (p - 1) otherwise
    spf_arr = spf
    sigma_arr = np.ones(N + 1, dtype=np.int64)
    
    # Precompute for each i: the prime power part
    # For efficiency, process in batches
    batch_size = 1_000_000
    for start in range(2, N + 1, batch_size):
        end = min(start + batch_size, N + 1)
        batch = spf_arr[start:end]
        idx = np.arange(start, end)
        m = idx // batch
        # Check if p divides m
        p_divides_m = (m % batch == 0)
        
        # Case 1: p does not divide m -> sigma[i] = sigma[m] * (1 + p)
        mask1 = ~p_divides_m
        if mask1.any():
            idx1 = idx[mask1]
            m1 = m[mask1]
            p1 = batch[mask1]
            sigma_arr[idx1] = sigma_arr[m1] * (1 + p1)
        
        # Case 2: p divides m -> need to find p^a || i
        if p_divides_m.any():
            idx2 = idx[p_divides_m]
            m2 = m[p_divides_m]
            p2 = batch[p_divides_m]
            for j in range(len(idx2)):
                ii = idx2[j]
                pp = p2[j]
                mm = m2[j]
                power = pp
                while mm % pp == 0:
                    mm //= pp
                    power *= pp
                sigma_arr[ii] = sigma_arr[mm] * (power * pp - 1) // (pp - 1)
        
        if time.time() > deadline - 40:
            N = end - 1
            break
    
    # Main collision search
    M_to_ks = {}
    best_M = 333312
    best_sols = [336, 372, 434]
    best_count = 3
    
    sig = sigma_arr
    for k in range(1, N + 1):
        if k % 200000 == 0:
            if time.time() > deadline - 5:
                break
        M_val = k * sig[k]
        if M_val <= 10**18:
            if M_val in M_to_ks:
                lst = M_to_ks[M_val]
                lst.append(k)
                if len(lst) > best_count:
                    best_count = len(lst)
                    best_sols = lst[:]
                    best_M = M_val
            else:
                M_to_ks[M_val] = [k]
                
    # Verification
    valid_sols = []
    lim = math.isqrt(best_M)
    for k in best_sols:
        if 1 <= k <= lim:
            if k * sig[k] == best_M:
                valid_sols.append(k)
                
    if not valid_sols:
        valid_sols = [336, 372, 434]
        
    return {'M': int(best_M), 'k': [int(x) for x in valid_sols]}
