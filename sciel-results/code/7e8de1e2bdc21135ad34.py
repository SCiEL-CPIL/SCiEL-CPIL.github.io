import time
import numpy as np
import random

_N = 256
_START_TIME = None
_DEADLINE = 58.0

def _thue_morse(n):
    signs = np.zeros(n, dtype=np.int64)
    for i in range(n):
        c = bin(i).count('1')
        signs[i] = 1 if c % 2 == 0 else -1
    return signs

def _legendre(n, p=257):
    signs = np.zeros(n, dtype=np.int64)
    for i in range(1, n+1):
        val = pow(i, (p-1)//2, p)
        if val == 1:
            signs[i-1] = 1
        else:
            signs[i-1] = -1
    return signs

def _fast_max_discrepancy(signs):
    n = len(signs)
    arr = np.asarray(signs, dtype=np.int64)
    max_disc = 1
    
    for d in range(1, n + 1):
        # Iterate over all residue classes modulo d
        for r in range(d):
            # Indices: r, r+d, r+2d, ... < n
            # Using slicing: arr[r::d]
            sub = arr[r::d]
            if len(sub) == 0:
                continue
            prefix = np.cumsum(sub)
            # Max absolute sum of any contiguous subarray in 'sub'
            # is max(prefix) - min(prefix)
            # Note: prefix[0] is sub[0]. The sum from 0 to j is prefix[j].
            # The sum from i to j is prefix[j] - prefix[i-1].
            # We need min prefix[i-1] and max prefix[j].
            # Let P = [0, prefix[0], prefix[1], ...]
            # Then max diff is max(P) - min(P).
            # np.cumsum gives [s0, s0+s1, ...]
            # We need to include 0 in the min/max calculation.
            min_val = np.min(prefix)
            max_val = np.max(prefix)
            # Wait, the sum can start at the beginning of the subarray.
            # So the "prefix before start" is 0.
            # So we should consider the set {0} U {prefix values}.
            # min_val = min(0, np.min(prefix))
            # max_val = max(0, np.max(prefix))
            # local_max = max_val - min_val
            
            # Optimization: 
            # If we just take np.min(prefix) and np.max(prefix), we miss the case where the min is 0 or max is 0.
            # Example: sub = [-1, -1]. prefix = [-1, -2]. min=-2, max=-1. diff=1.
            # But sum of [-1] is -1. abs=1. Sum of [-1,-1] is -2. abs=2.
            # Wait. 
            # Sums: 
            # i=0, j=0: -1
            # i=0, j=1: -2
            # i=1, j=1: -1
            # Max abs is 2.
            # My formula: max(P) - min(P) where P includes 0?
            # P = [0, -1, -2]. max=0, min=-2. diff=2. Correct.
            
            # So we must include 0.
            # Instead of creating a new array, we can just check if 0 is outside [min, max].
            p_min = np.min(prefix)
            p_max = np.max(prefix)
            
            if p_min > 0:
                p_min = 0
            if p_max < 0:
                p_max = 0
            
            local_max = p_max - p_min
            if local_max > max_disc:
                max_disc = int(local_max)
                
    return max_disc

def _local_search(signs, time_limit, seed):
    n = len(signs)
    signs = np.array(signs, dtype=np.int64)
    current_disc = _fast_max_discrepancy(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    rng = random.Random(seed)
    T = 5.0
    alpha = 0.999
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        idx = rng.randint(0, n - 1)
        old_sign = signs[idx]
        signs[idx] = -old_sign
        
        new_disc = _fast_max_discrepancy(signs)
        delta = new_disc - current_disc
        
        if delta < 0 or (T > 0.01 and rng.random() < np.exp(-delta / T)):
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs.copy()
        else:
            signs[idx] = old_sign
            
        T *= alpha
        
    return best_signs.tolist(), best_disc

def construct():
    global _START_TIME, _DEADLINE
    n = _N
    _START_TIME = time.time()
    
    best_signs = None
    best_disc = float('inf')
    
    # Generate diverse initializations
    inits = []
    inits.append(_thue_morse(n))
    inits.append(_legendre(n))
    
    # Add some random inits
    for s in [1, 2, 3]:
        rs = np.random.RandomState(s)
        inits.append(rs.choice([-1, 1], n))
        
    # Add an alternating init
    alt = np.zeros(n, dtype=np.int64)
    for i in range(n):
        alt[i] = 1 if i % 2 == 0 else -1
    inits.append(alt)
    
    # Time budget per run
    # Total budget ~ 58s. 5 inits. ~11s each.
    # But we can stop early if we find a very good solution?
    # Or just distribute time.
    
    time_per_run = 11.0
    
    for i, init_signs in enumerate(inits):
        if time.time() - _START_TIME > _DEADLINE:
            break
            
        # Adjust time limit for the last run
        remaining = _DEADLINE - (time.time() - _START_TIME)
        current_time_limit = min(time_per_run, remaining)
        
        if current_time_limit <= 0:
            break

        res_signs, res_disc = _local_search(init_signs, current_time_limit, seed=100+i)
        
        if res_disc < best_disc:
            best_disc = res_disc
            best_signs = res_signs
            
    if best_signs is None:
        return [1] * n
        
    return best_signs
