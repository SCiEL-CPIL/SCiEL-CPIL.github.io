import random
import time
import math
import numpy as np

def _build_ap_structure(n):
    """Precompute all arithmetic progressions and their element indices."""
    aps = []
    for d in range(1, n):
        for r in range(d):
            idxs = list(range(r, n, d))
            if len(idxs) > 1:
                aps.append(idxs)
    return aps

def _calc_max_disc(signs, aps, limit=None):
    """Calculate max AP discrepancy with precomputed structure."""
    max_disc = 1
    for idxs in aps:
        curr = 0
        min_p = 0
        max_p = 0
        for idx in idxs:
            curr += signs[idx]
            if curr < min_p:
                min_p = curr
            if curr > max_p:
                max_p = curr
        disc = max_p - min_p
        if disc > max_disc:
            max_disc = disc
            if limit is not None and max_disc > limit:
                return max_disc
    return max_disc

def _greedy_repair(signs, aps, max_iters=1000, time_limit=None, start_time=None):
    """Greedy repair: flip the element in the worst AP that reduces discrepancy most."""
    n = len(signs)
    current_disc = _calc_max_disc(signs, aps)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Find worst AP
        worst_disc = 0
        worst_ap = None
        for idxs in aps:
            curr = 0
            min_p = 0
            max_p = 0
            for idx in idxs:
                curr += signs[idx]
                if curr < min_p:
                    min_p = curr
                if curr > max_p:
                    max_p = curr
            disc = max_p - min_p
            if disc > worst_disc:
                worst_disc = disc
                worst_ap = idxs
        
        if worst_disc <= 1:
            break
        
        # Try flipping each element in worst AP
        best_flip_idx = None
        best_new_disc = current_disc
        for idx in worst_ap:
            signs[idx] *= -1
            new_disc = _calc_max_disc(signs, aps, limit=current_disc)
            signs[idx] *= -1
            if new_disc < best_new_disc:
                best_new_disc = new_disc
                best_flip_idx = idx
        
        if best_flip_idx is not None:
            signs[best_flip_idx] *= -1
            current_disc = best_new_disc
        else:
            break
    
    return current_disc

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # 55s margin
    
    # Build AP structure once
    aps = _build_ap_structure(n)
    
    # Initial solution
    signs = [random.choice((-1, 1)) for _ in range(n)]
    current_disc = _calc_max_disc(signs, aps)
    best_disc = current_disc
    best_signs = signs[:]
    
    # Simulated Annealing parameters
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() < deadline:
        # Pick a random index to flip
        i = random.randint(0, n - 1)
        signs[i] *= -1
        
        # Calculate new discrepancy with a limit to speed up rejection
        limit = current_disc + int(T * 3) + 1
        new_disc = _calc_max_disc(signs, aps, limit)
        
        delta = new_disc - current_disc
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_disc = new_disc
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs[:]
        else:
            # Reject
            signs[i] *= -1
            
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Periodic greedy repair every ~5000 iterations
        if random.random() < 0.0002:
            current_disc = _greedy_repair(signs, aps, max_iters=200, 
                                         time_limit=2.0, start_time=time.time())
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs[:]
    
    # Final greedy repair
    current_disc = _greedy_repair(signs, aps, max_iters=500,
                                  time_limit=5.0, start_time=time.time())
    if current_disc < best_disc:
        best_disc = current_disc
        best_signs = signs[:]
    
    return best_signs
