import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 80
    N = 10000000
    
    # Compute sigma(n) using linear sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    primes = []
    is_composite = np.zeros(N + 1, dtype=bool)
    lp = np.zeros(N + 1, dtype=np.int64)
    sigma_lp = np.zeros(N + 1, dtype=np.int64)
    
    lp[1] = 1
    sigma_lp[1] = 1
    sigma[1] = 1
    
    for i in range(2, N + 1):
        if time.time() > deadline - 5:
            break
        if not is_composite[i]:
            primes.append(i)
            lp[i] = i
            sigma_lp[i] = 1 + i
            sigma[i] = 1 + i
        for p in primes:
            prod = i * p
            if prod > N:
                break
            is_composite[prod] = True
            if i % p == 0:
                lp[prod] = lp[i] * p
                sigma_lp[prod] = sigma_lp[i] + lp[i] * p
                sigma[prod] = sigma[i] // sigma_lp[i] * sigma_lp[prod]
                break
            else:
                lp[prod] = p
                sigma_lp[prod] = 1 + p
                sigma[prod] = sigma[i] * (1 + p)
    
    if time.time() > deadline - 5:
        return [11, 6]
    
    # Sort by sigma value
    order = np.argsort(sigma[1:], kind='stable')
    sorted_sigma = sigma[1:][order]
    sorted_n = order + 1
    
    best_err = float('inf')
    best_pair = (11, 6)
    
    i = 0
    while i < N:
        if time.time() > deadline - 10:
            break
        j = i + 1
        while j < N and sorted_sigma[j] == sorted_sigma[i]:
            j += 1
        if j - i >= 2:
            group = sorted_n[i:j]
            for ia in range(len(group)):
                a_val = int(group[ia])
                target = 2.0 * a_val / 3.0
                pos = np.searchsorted(group, target)
                for p in [pos - 1, pos, pos + 1]:
                    if 0 <= p < len(group):
                        b_val = int(group[p])
                        if a_val != b_val:
                            err = abs(2 * a_val - 3 * b_val) / (2 * b_val)
                            if err < best_err:
                                best_err = err
                                best_pair = (a_val, b_val)
        i = j
    
    return [int(best_pair[0]), int(best_pair[1])]
