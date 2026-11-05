import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 20s safety margin

    def generate_powerful(limit):
        powerful = set([1])
        max_b = int(round(limit ** (1/3))) + 1
        
        for b in range(1, max_b + 1):
            is_sq_free = True
            temp_b = b
            d = 2
            while d * d <= temp_b:
                if temp_b % (d * d) == 0:
                    is_sq_free = False
                    break
                d += 1
            if not is_sq_free:
                continue
                
            b_cubed = b ** 3
            if b_cubed > limit:
                break
                
            max_a = isqrt(limit // b_cubed)
            for a in range(1, max_a + 1):
                val = a * a * b_cubed
                if val <= limit:
                    powerful.add(val)
                else:
                    break
                    
        return sorted(powerful)

    def find_best_in_limit(limit):
        powerful_list = generate_powerful(limit)
        powerful_set = set(powerful_list)
        
        S = np.zeros(limit + 1, dtype=np.int8)
        for p in powerful_list:
            S[p] = 1
            
        try:
            conv = np.convolve(S, S, 'full')
        except MemoryError:
            return None, None, None
            
        max_val = np.max(conv)
        candidates = np.where(conv == max_val)[0]
        
        best_N = -1
        best_count = -1
        best_pairs = []
        
        for N in candidates:
            N_int = int(N)
            if N_int < 2:
                continue
            c = conv[N_int]
            d = 1 if (N_int % 2 == 0 and (N_int // 2) in powerful_set) else 0
            count = (c + d) // 2
            if count > best_count:
                best_count = count
                best_N = N_int
                
        if best_N == -1:
            return None, None, None
            
        pairs = []
        for a in powerful_list:
            if 2 * a > best_N:
                break
            b = best_N - a
            if b in powerful_set:
                pairs.append([a, b])
                
        return best_N, best_count, pairs

    best_N = 2
    best_pairs = [[1, 1]]
    best_score = 1
    
    limits = [10**6, 10**7, 10**8]
    
    for limit in limits:
        if time.time() > deadline - 20:
            break
            
        try:
            N, count, pairs = find_best_in_limit(limit)
            if N is not None and count > best_score:
                best_N = N
                best_pairs = pairs
                best_score = count
        except Exception:
            pass
            
    return {'N': best_N, 'pairs': best_pairs}
