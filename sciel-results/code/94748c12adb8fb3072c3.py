import time
import numpy as np
import bisect

def construct():
    t0 = time.time()
    HARD_STOP = t0 + 55.0 # Safety margin
    
    N = 10000000
    
    # Linear Sieve for Sigma
    sigma = [0] * (N + 1)
    sigma_pow = [0] * (N + 1)
    cnt = [0] * (N + 1)
    spf = [0] * (N + 1)
    p_power = [0] * (N + 1)
    
    sigma[1] = 1
    sigma_pow[1] = 1
    cnt[1] = 0
    spf[1] = 1
    p_power[1] = 1
    
    primes = []
    
    for i in range(2, N + 1):
        if time.time() > HARD_STOP - 10.0:
            # Fallback if sieve is too slow
            return [11, 6]
            
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            sigma[i] = 1 + i
            sigma_pow[i] = 1 + i
            cnt[i] = 1
            p_power[i] = i
            
        for p in primes:
            ip = i * p
            if ip > N:
                break
            spf[ip] = p
            if p == spf[i]:
                cnt[ip] = cnt[i] + 1
                p_power[ip] = p_power[i] * p
                sigma_pow[ip] = sigma_pow[i] + p_power[ip]
                sigma[ip] = sigma[i] // sigma_pow[i] * sigma_pow[ip]
                break
            else:
                cnt[ip] = 1
                p_power[ip] = p
                sigma_pow[ip] = 1 + p
                sigma[ip] = sigma[i] * (1 + p)
                
    if time.time() > HARD_STOP - 5.0:
        return [11, 6]
        
    # Convert to NumPy array
    sigma_arr = np.array(sigma[1:], dtype=np.int64)
    
    # Sort indices by sigma value
    order = np.argsort(sigma_arr, kind='stable')
    sorted_sigma = sigma_arr[order]
    sorted_indices = order + 1
    
    # Find group boundaries
    changes = np.nonzero(sorted_sigma[1:] != sorted_sigma[:-1])[0]
    starts = np.concatenate(([0], changes + 1))
    ends = np.concatenate((changes + 1, [len(sorted_sigma)]))
    
    best_val = float('inf')
    best_pair = None
    
    for gi in range(len(starts)):
        if time.time() > HARD_STOP - 2.0:
            break
            
        s0 = starts[gi]
        e0 = ends[gi]
        length = e0 - s0
        
        if length < 2:
            continue
            
        # Extract group and sort by value
        gn = sorted_indices[s0:e0]
        gn = np.sort(gn)
        L = len(gn)
        
        # Convert to list for faster bisect if L is small?
        # bisect on numpy array is slower than on list?
        # Let's try list.
        gn_list = gn.tolist()
        
        for bi in range(L):
            b = gn_list[bi]
            target = 1.5 * b
            
            # Binary search for target in gn_list
            j = bisect.bisect_left(gn_list, target)
            
            # Check right neighbor
            if j < L:
                a = gn_list[j]
                if a != b:
                    num = abs(2 * a - 3 * b)
                    if num > 0:
                        val = num / (2.0 * b)
                        if val < best_val:
                            best_val = val
                            best_pair = (a, b)
            
            # Check left neighbor
            if j > 0:
                a = gn_list[j - 1]
                if a != b:
                    num = abs(2 * a - 3 * b)
                    if num > 0:
                        val = num / (2.0 * b)
                        if val < best_val:
                            best_val = val
                            best_pair = (a, b)
                            
    if best_pair is None:
        return [11, 6]
        
    a, b = best_pair
    # Verify
    if sigma[a] != sigma[b]:
        return [11, 6]
    if 2 * a == 3 * b:
        return [11, 6]
        
    return [int(a), int(b)]
