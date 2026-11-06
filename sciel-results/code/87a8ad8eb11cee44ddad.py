"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import heapq

def construct():
    # Generate primes <= 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # Window size
    W = 10000
    
    # Initialize residues
    residues = np.ones(n_primes, dtype=np.int64)
    
    # Initialize coverage
    cov = np.zeros(W + 1, dtype=np.int32)
    for i in range(n_primes):
        p = primes_arr[i]
        r = residues[i]
        start = r if r >= 1 else p
        if start <= W:
            cov[start:W+1:p] += 1
            
    # Initialize bad set (heap) and is_bad array
    is_bad = np.zeros(W + 1, dtype=bool)
    bad_heap = []
    
    for x in range(1, W + 1):
        if cov[x] < 2:
            is_bad[x] = True
            heapq.heappush(bad_heap, x)
            
    def get_L():
        while bad_heap:
            x = bad_heap[0]
            if is_bad[x]:
                return x - 1
            else:
                heapq.heappop(bad_heap)
        return W
        
    current_L = get_L()
    best_L = current_L
    best_residues = residues.copy()
    
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    start_time = time.time()
    time_limit = 100
    
    random.seed(42)
    
    while time.time() - start_time < time_limit and T > T_min:
        i = random.randint(0, n_primes - 1)
        p = primes_arr[i]
        old_r = residues[i]
        
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
            
        # Update coverage: Remove old
        old_start = old_r if old_r >= 1 else p
        if old_start <= W:
            indices = np.arange(old_start, W + 1, p)
            c_old = cov[indices]
            cov[indices] -= 1
            to_bad = indices[c_old == 2]
            if len(to_bad) > 0:
                is_bad[to_bad] = True
                bad_heap.extend(to_bad.tolist())
                heapq.heapify(bad_heap)
                
        # Update coverage: Add new
        new_start = new_r if new_r >= 1 else p
        if new_start <= W:
            indices = np.arange(new_start, W + 1, p)
            c_old = cov[indices]
            cov[indices] += 1
            to_good = indices[c_old == 1]
            if len(to_good) > 0:
                is_bad[to_good] = False
        
        new_L = get_L()
        
        delta_L = new_L - current_L
        if delta_L >= 0:
            current_L = new_L
            residues[i] = new_r
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
        else:
            if random.random() < np.exp(delta_L / T):
                current_L = new_L
                residues[i] = new_r
            else:
                # Revert
                if new_start <= W:
                    indices = np.arange(new_start, W + 1, p)
                    c_old = cov[indices]
                    cov[indices] -= 1
                    to_bad = indices[c_old == 2]
                    if len(to_bad) > 0:
                        is_bad[to_bad] = True
                        bad_heap.extend(to_bad.tolist())
                        heapq.heapify(bad_heap)
                        
                if old_start <= W:
                    indices = np.arange(old_start, W + 1, p)
                    c_old = cov[indices]
                    cov[indices] += 1
                    to_good = indices[c_old == 1]
                    if len(to_good) > 0:
                        is_bad[to_good] = False
                        
        T *= alpha
        
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
