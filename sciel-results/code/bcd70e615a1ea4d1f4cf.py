import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    primes_1_4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193]
    
    best_score = 0
    best_result = {'N': 2, 'pairs': [[1, 1]]}
    
    # Use a priority queue or just keep top K candidates to avoid memory blowup
    # But simple list is fine if we prune.
    
    candidates = []
    
    def backtrack_v2(idx, current_n, current_divs):
        if time.time() > deadline - 10:
            return
        if idx == len(primes_1_4):
            if current_n > 1 and current_n <= 10**18:
                candidates.append((current_n, current_divs))
            return
        
        p = primes_1_4[idx]
        power = 1
        e = 1
        while current_n * power <= 10**18:
            backtrack_v2(idx + 1, current_n * power, current_divs * (e + 1))
            power *= p
            e += 1
            if time.time() > deadline - 10:
                return
    
    backtrack_v2(0, 1, 1)
    
    # Find best N
    best_n = 1
    best_divs = 0
    for n, d in candidates:
        if d > best_divs:
            best_divs = d
            best_n = n
    
    # Find pairs for best_n
    max_x = int(math.isqrt(best_n // 2))
    
    pairs = []
    chunk_size = 10**6
    for start in range(1, max_x + 1, chunk_size):
        end = min(start + chunk_size, max_x + 1)
        x = np.arange(start, end, dtype=np.int64)
        y2 = best_n - x*x
        
        # Safe square check
        y_int = np.sqrt(y2.astype(np.float64)).astype(np.int64)
        is_square = (y_int * y_int == y2) | ((y_int + 1) * (y_int + 1) == y2) | ((y_int - 1) * (y_int - 1) == y2)
        
        idx = np.where(is_square)[0]
        for i in idx:
            a = int(x[i]) ** 2
            b = best_n - a
            pairs.append([a, b])
        
        if time.time() > deadline - 5:
            break
    
    if len(pairs) > best_score:
        best_score = len(pairs)
        best_result = {'N': best_n, 'pairs': pairs}
    
    return best_result
