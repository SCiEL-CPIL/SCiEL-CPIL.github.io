import time

P = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def smooth_part(x):
    p = 1
    for prime in P:
        while x % prime == 0:
            p *= prime
            x //= prime
    return p

def construct():
    best_n = 1000
    best_score = float('inf')
    
    start_time = time.time()
    
    # Scan n from 1000 to 1000000
    # We can step by 1.
    # 10^6 * 24 operations is ~24 million. Python might be slow.
    # 24 million simple operations might take a few seconds.
    # Let's optimize the inner loop.
    
    for n in range(1000, 1000001):
        # Check time
        if time.time() - start_time > 45: # Safety margin
            break
        
        factors = set()
        # Unroll or just loop
        for i in range(1, 25):
            factors.add(smooth_part(n + i))
        
        score = len(factors)
        if score < best_score:
            best_score = score
            best_n = n
            if best_score == 2:
                # Cannot do better than 2
                break
                
    return best_n
