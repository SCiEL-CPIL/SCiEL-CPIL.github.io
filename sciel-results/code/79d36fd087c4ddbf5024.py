import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Try to find consecutive P-smooth numbers for small P
    # This could yield a much better score if such pairs exist
    # Check P=5, 7, 11, 13, 17, 19, 23
    
    def find_smooth_pair(P):
        # Generate all P-smooth numbers <= N_MAX + 1
        primes = []
        for i in range(2, P+1):
            # Check if i is prime
            is_p = True
            for j in range(2, int(i**0.5)+1):
                if i % j == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(i)
                
        if not primes:
            return None
            
        # Generate smooth numbers using a heap-based approach
        # This correctly generates all P-smooth numbers
        import heapq
        
        smooth_nums = set([1])
        heap = [1]
        
        while heap:
            val = heapq.heappop(heap)
            for p in primes:
                next_val = val * p
                if next_val <= N_MAX + 1 and next_val not in smooth_nums:
                    smooth_nums.add(next_val)
                    heapq.heappush(heap, next_val)
        
        # Convert to sorted list
        candidates = sorted(smooth_nums)
        
        # Check for consecutive
        for i in range(len(candidates) - 1):
            if candidates[i+1] - candidates[i] == 1:
                n = candidates[i]
                if n >= N_MIN:
                    return n
        return None

    # Try P=5
    n_found = find_smooth_pair(5)
    if n_found is not None:
        return n_found
        
    # Try P=7
    n_found = find_smooth_pair(7)
    if n_found is not None:
        return n_found
        
    # Try P=11
    n_found = find_smooth_pair(11)
    if n_found is not None:
        return n_found
        
    # Try P=13
    n_found = find_smooth_pair(13)
    if n_found is not None:
        return n_found
        
    # Try P=17
    n_found = find_smooth_pair(17)
    if n_found is not None:
        return n_found
        
    # Try P=19
    n_found = find_smooth_pair(19)
    if n_found is not None:
        return n_found
        
    # Try P=23
    n_found = find_smooth_pair(23)
    if n_found is not None:
        return n_found
        
    # If no small P pair found, fall back to full scan for min LPF
    # This will find the global minimum, likely 13
    
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
