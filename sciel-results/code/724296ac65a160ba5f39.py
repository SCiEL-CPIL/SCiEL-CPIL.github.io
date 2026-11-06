import numpy as np
import time
import random

def construct():
    n = 20
    max_val = 10000
    start_time = time.time()
    time_limit = 85.0
    
    def get_metrics(A):
        sums = np.add.outer(A, A)
        prods = np.multiply.outer(A, A)
        s_size = len(np.unique(sums))
        p_size = len(np.unique(prods))
        return s_size, p_size

    best_val = float('inf')
    best_A = None
    
    # Try a few different initializations
    inits = [
        np.array(list(range(1, n + 1)), dtype=np.int64), # AP
        np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 3, 6, 12, 24, 48, 96], dtype=np.int64), # Mixed
        np.array(sorted(random.sample(range(1, max_val), n)), dtype=np.int64) # Random
    ]
    
    for init_A in inits:
        if time.time() - start_time > time_limit - 5:
            break
            
        A = init_A.copy()
        s_size, p_size = get_metrics(A)
        current_val = max(s_size, p_size)
        
        if current_val < best_val:
            best_val = current_val
            best_A = A.copy()
            
        stagnant_count = 0
        
        while time.time() - start_time < time_limit - 2:
            idx = random.randint(0, n - 1)
            old_val = A[idx]
            
            # Generate candidate
            # 80% random, 20% local
            if random.random() < 0.8:
                new_val = random.randint(1, max_val)
            else:
                new_val = old_val + random.randint(-10, 10)
                if new_val < 1: new_val = 1
                if new_val > max_val: new_val = max_val
                
            if new_val in A:
                continue
                
            A_new = A.copy()
            A_new[idx] = new_val
            
            s_new, p_new = get_metrics(A_new)
            val = max(s_new, p_new)
            
            if val < current_val:
                current_val = val
                A = A_new
                stagnant_count = 0
                if val < best_val:
                    best_val = val
                    best_A = A.copy()
            else:
                stagnant_count += 1
                if stagnant_count > 5000:
                    break # Restart this initialization
            
    return list(best_A)
