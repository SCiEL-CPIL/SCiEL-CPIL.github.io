import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N_MIN = 10000
    N_MAX = 1000000
    
    def find_smooth_pair(P):
        # Get primes up to P
        primes = []
        for i in range(2, P+1):
            is_p = True
            for j in range(2, int(i**0.5)+1):
                if i % j == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(i)
                
        if not primes:
            return None
            
        smooth_nums = set([1])
        changed = True
        while changed:
            changed = False
            current_list = list(smooth_nums)
            for num in current_list:
                for p in primes:
                    new_num = num * p
                    if new_num <= N_MAX + 1 and new_num not in smooth_nums:
                        smooth_nums.add(new_num)
                        changed = True
        
        candidates = sorted(smooth_nums)
        
        for i in range(len(candidates) - 1):
            if candidates[i+1] - candidates[i] == 1:
                n = candidates[i]
                if n >= N_MIN and n <= N_MAX:
                    return n
        return None

    # Try small P values
    for P in [5, 7, 11]:
        if time.time() > deadline - 10:
            break
        n_found = find_smooth_pair(P)
        if n_found is not None:
            return n_found
            
    # Fallback to sieve
    limit = N_MAX + 1
    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i : limit+1 : i] = False
    
    primes = np.where(is_prime)[0]
    
    lpf = np.zeros(limit + 1, dtype=np.int64)
    for p in primes:
        multiples = np.arange(p, limit + 1, p)
        lpf[multiples] = p
    
    if time.time() > deadline - 5:
        return 10000
        
    n_range = np.arange(N_MIN, N_MAX + 1)
    lpf_n = lpf[n_range]
    lpf_np1 = lpf[n_range + 1]
    scores = np.maximum(lpf_n, lpf_np1)
    
    best_idx = np.argmin(scores)
    best_n = int(n_range[best_idx])
    
    return best_n
