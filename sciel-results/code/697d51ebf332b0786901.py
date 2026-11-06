import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140.0
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, p in enumerate(is_prime) if p]
    
    primes = sieve(1000)[:168]
    n_primes = len(primes)
    LIMIT = 5000
    
    r = np.array([random.randint(0, p-1) for p in primes], dtype=int)
    cover_count = np.zeros(LIMIT + 1, dtype=int)
    
    for i, p in enumerate(primes):
        ri = r[i]
        start = p if ri == 0 else ri
        if start <= LIMIT:
            cover_count[start:LIMIT+1:p] += 1
            
    def get_L():
        # Find first x in 1..LIMIT where cover_count[x] < 2
        # Using numpy argmin on a boolean mask? 
        # Or just a loop? 
        # Loop is fine for 5000.
        for x in range(1, LIMIT + 1):
            if cover_count[x] < 2:
                return x - 1
        return LIMIT

    current_L = get_L()
    best_L = current_L
    best_r = r.copy()
    
    def get_defects():
        return np.sum(cover_count[1:LIMIT+1] < 2)
        
    current_defects = get_defects()
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    accepted_count = 0
    
    while time.time() < deadline:
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = r[i]
        new_r = random.randint(0, p - 1)
        
        if old_r == new_r:
            continue
            
        old_start = p if old_r == 0 else old_r
        new_start = p if new_r == 0 else new_r
        
        # Apply changes
        if old_start <= LIMIT:
            cover_count[old_start:LIMIT+1:p] -= 1
        if new_start <= LIMIT:
            cover_count[new_start:LIMIT+1:p] += 1
            
        new_defects = get_defects()
        delta = new_defects - current_defects
        
        accepted = False
        if delta < 0:
            accepted = True
        else:
            if delta == 0:
                accepted = random.random() < 0.5
            else:
                if T > 0:
                    prob = np.exp(-delta / T)
                else:
                    prob = 0
                accepted = random.random() < prob
        
        if accepted:
            r[i] = new_r
            current_defects = new_defects
            accepted_count += 1
            
            # Check L periodically
            if accepted_count % 50 == 0:
                L = get_L()
                if L > best_L:
                    best_L = L
                    best_r = r.copy()
        else:
            # Revert
            if old_start <= LIMIT:
                cover_count[old_start:LIMIT+1:p] += 1
            if new_start <= LIMIT:
                cover_count[new_start:LIMIT+1:p] -= 1
                
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_r.tolist()
