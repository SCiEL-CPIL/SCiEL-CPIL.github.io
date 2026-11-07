import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    N = 10_000_000
    
    # Multiplicative sieve for sigma(n)
    # sigma is multiplicative: sigma(n) = prod_p (1 + p + p^2 + ... + p^k)
    sigma = np.ones(N + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes to find primes
    is_prime = np.ones(N + 1, dtype=np.bool_)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # For each prime p, apply sigma factor for each prime power p^k
    for p in primes:
        if time.time() - start_time > DEADLINE - 10:
            break
        p = int(p)
        pk = p
        while pk <= N:
            factor = (pk * p - 1) // (p - 1)  # 1 + p + ... + p^k
            pkp = pk * p
            if pkp <= N:
                sigma[pk::pk] *= factor
                sigma[pkp::pkp] //= factor
            else:
                sigma[pk::pk] *= factor
            pk = pkp
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return [11, 6]
    
    # Group indices by sigma value
    sig = sigma[1:N+1]
    
    order = np.argsort(sig, kind='quicksort')
    sorted_sig = sig[order]
    sorted_idx = order + 1
    
    changes = np.diff(sorted_sig) != 0
    boundaries = np.concatenate(([0], np.where(changes)[0] + 1, [N]))
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    num_groups = len(boundaries) - 1
    
    for gi in range(num_groups):
        if time.time() - start_time > DEADLINE - 3:
            break
        s = boundaries[gi]
        e = boundaries[gi+1]
        size = e - s
        if size <= 1:
            continue
        
        group = sorted_idx[s:e].copy()
        group.sort()
        
        targets = (3 * group.astype(np.float64)) / 2.0
        pos = np.searchsorted(group, targets, side='left')
        size_int = len(group)
        
        mask1 = pos < size_int
        if np.any(mask1):
            p1 = pos[mask1]
            a_cand = group[p1]
            b_cand = group[mask1]
            valid = a_cand != b_cand
            if np.any(valid):
                a_v = a_cand[valid]
                b_v = b_cand[valid]
                num = np.abs(2 * a_v.astype(np.int64) - 3 * b_v.astype(np.int64))
                den = 2 * b_v.astype(np.int64)
                errs = num.astype(np.float64) / den.astype(np.float64)
                min_i = np.argmin(errs)
                err_val = errs[min_i]
                if err_val < best_error:
                    best_error = err_val
                    best_a = int(a_v[min_i])
                    best_b = int(b_v[min_i])
        
        mask2 = pos > 0
        if np.any(mask2):
            p2 = pos[mask2] - 1
            a_cand = group[p2]
            b_cand = group[mask2]
            valid = a_cand != b_cand
            if np.any(valid):
                a_v = a_cand[valid]
                b_v = b_cand[valid]
                num = np.abs(2 * a_v.astype(np.int64) - 3 * b_v.astype(np.int64))
                den = 2 * b_v.astype(np.int64)
                errs = num.astype(np.float64) / den.astype(np.float64)
                min_i = np.argmin(errs)
                err_val = errs[min_i]
                if err_val < best_error:
                    best_error = err_val
                    best_a = int(a_v[min_i])
                    best_b = int(b_v[min_i])
    
    return [int(best_a), int(best_b)]
