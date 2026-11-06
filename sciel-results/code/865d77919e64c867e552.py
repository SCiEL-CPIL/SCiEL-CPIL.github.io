# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import heapq
import time

def construct():
    # Generate primes <= 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # Window size. Let's aim for 10000 to allow finding L > 1000 easily.
    W = 10000
    
    # Initialize residues
    # Heuristic: r_p = 1
    residues = np.ones(n_primes, dtype=np.int64)
    
    # Compute initial coverage
    cov = np.zeros(W + 1, dtype=np.int32)
    for i in range(n_primes):
        p = primes_arr[i]
        r = residues[i]
        start = r if r >= 1 else p
        if start <= W:
            cov[start:W+1:p] += 1
            
    # Initialize heap of bad positions (where cov < 2)
    bad_heap = []
    for x in range(1, W + 1):
        if cov[x] < 2:
            heapq.heappush(bad_heap, x)
            
    def get_L():
        # Clean heap top
        while bad_heap:
            x = bad_heap[0]
            if cov[x] >= 2:
                heapq.heappop(bad_heap)
            else:
                return x - 1
        return W
        
    current_L = get_L()
    best_L = current_L
    best_residues = residues.copy()
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    max_iters = 50000
    random.seed(42)
    
    for it in range(max_iters):
        # Pick random prime
        i = random.randint(0, n_primes - 1)
        p = primes_arr[i]
        old_r = residues[i]
        
        # Pick new residue
        if random.random() < 0.5:
            new_r = random.randint(0, p - 1)
        else:
            # Try to cover current L+1 or nearby
            target = min(W, current_L + 1)
            new_r = target % p
            if new_r == old_r:
                new_r = (new_r + 1) % p
                
        if new_r == old_r:
            continue
            
        # Update coverage
        old_start = old_r if old_r >= 1 else p
        new_start = new_r if new_r >= 1 else p
        
        affected = []
        
        # Decrement old
        if old_start <= W:
            for x in range(old_start, W + 1, p):
                cov[x] -= 1
                if cov[x] < 2:
                    heapq.heappush(bad_heap, x)
                affected.append(x)
                
        # Increment new
        if new_start <= W:
            for x in range(new_start, W + 1, p):
                cov[x] += 1
                # If it was bad, it might be fixed. We don't remove from heap, 
                # get_L() will handle stale entries.
                affected.append(x)
                
        new_L = get_L()
        
        # Acceptance criterion
        delta = new_L - current_L
        if delta > 0 or random.random() < math.exp(delta / T):
            residues[i] = new_r
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
                # Optional: reheat if we found a new best
                # T = max(T_min, T * 0.9) 
        else:
            # Revert coverage
            if old_start <= W:
                for x in range(old_start, W + 1, p):
                    cov[x] += 1
            if new_start <= W:
                for x in range(new_start, W + 1, p):
                    cov[x] -= 1
                    # If it becomes bad, push to heap
                    if cov[x] < 2:
                        heapq.heappush(bad_heap, x)
                    
        T *= cooling
        if T < T_min:
            T = T_min
            
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
