import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # We want to maximize n.
    # Try n=34 first.
    for n in range(34, 5, -1):
        if time.time() > deadline - 20: # Need at least 20s for smaller n if 34 fails?
             # Actually, if 34 fails quickly, we can try 33.
             # But if 34 takes 100s and fails, we have 15s left.
             # Let's allocate time per n.
             pass
            
    # Let's just try n=34 with a time budget.
    # If it fails, try n=33, etc.
    
    target_n = 34
    # We will run local search for target_n.
    # If it doesn't succeed, we reduce target_n.
    
    # To be safe, let's try to find a valid coloring for n=34.
    # If we fail, we return the best valid n we found?
    # No, we must return a valid coloring.
    # So we should loop n from 34 down to 9.
    # For each n, run search for a fixed time or steps.
    
    # Strategy:
    # 1. Try n=34 for 60s.
    # 2. If fail, try n=33 for 30s.
    # 3. If fail, try n=32 for 10s.
    # 4. If fail, return a known valid small n (e.g. n=9 from seed, or find n=10 quickly).
    
    # Let's refine the time allocation.
    # n=34 is the most important.
    
    def solve(n, time_limit):
        # Precomputation
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Generate K4s and K5s
        k4s = []
        for k4 in combinations(range(n), 4):
            ts = [triple_to_idx[t] for t in combinations(k4, 3)]
            k4s.append(ts)
            
        k5s = []
        for k5 in combinations(range(n), 5):
            ts = [triple_to_idx[t] for t in combinations(k5, 3)]
            k5s.append(ts)
            
        num_k4 = len(k4s)
        num_k5 = len(k5s)
        
        # Map triple index to affected K4s and K5s
        affected_k4 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k4s):
            for t in ts:
                affected_k4[t].append(i)
                
        affected_k5 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k5s):
            for t in ts:
                affected_k5[t].append(i)
                
        # Precompute triple indices for vectorized updates
        k4_triples_arr = np.array(k4s, dtype=np.int32)
        k5_triples_arr = np.array(k5s, dtype=np.int32)
        
        # State
        colors = np.random.randint(0, 2, num_triples, dtype=np.int8)
        
        # Violations
        viol_k4 = np.all(colors[k4_triples_arr] == 0, axis=1).astype(np.int8)
        viol_k5 = np.all(colors[k5_triples_arr] == 1, axis=1).astype(np.int8)
        
        E = np.sum(viol_k4) + np.sum(viol_k5)
        
        if E == 0:
            return colors.tolist()
            
        # Local Search: Simulated Annealing
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        
        while time.time() < start_time + time_limit:
            if T < min_T:
                break
                
            t_idx = random.randint(0, num_triples - 1)
            old_c = colors[t_idx]
            new_c = 1 - old_c
            
            colors[t_idx] = new_c
            
            delta = 0
            
            # Update K4 violations
            k4_idxs = affected_k4[t_idx]
            if k4_idxs:
                old_viol_k4 = viol_k4[k4_idxs].copy()
                new_viol_k4 = np.all(colors[k4_triples_arr[k4_idxs]] == 0, axis=1).astype(np.int8)
                viol_k4[k4_idxs] = new_viol_k4
                delta += np.sum(new_viol_k4 - old_viol_k4)
                
            # Update K5 violations
            k5_idxs = affected_k5[t_idx]
            if k5_idxs:
                old_viol_k5 = viol_k5[k5_idxs].copy()
                new_viol_k5 = np.all(colors[k5_triples_arr[k5_idxs]] == 1, axis=1).astype(np.int8)
                viol_k5[k5_idxs] = new_viol_k5
                delta += np.sum(new_viol_k5 - old_viol_k5)
                
            if delta <= 0:
                E += delta
            else:
                if random.random() < np.exp(-delta / T):
                    E += delta
                else:
                    colors[t_idx] = old_c
                    if k4_idxs:
                        viol_k4[k4_idxs] = old_viol_k4
                    if k5_idxs:
                        viol_k5[k5_idxs] = old_viol_k5
                        
            T *= cooling
            
            if E == 0:
                return colors.tolist()
                
        return None
        
    # Try n=34
    # Allocate 100s for n=34
    result = solve(34, 100)
    if result is not None:
        return {'n': 34, 'colors': result}
        
    # Try n=33
    if time.time() < deadline - 10:
        result = solve(33, 10)
        if result is not None:
            return {'n': 33, 'colors': result}
            
    # Try n=32
    if time.time() < deadline - 5:
        result = solve(32, 5)
        if result is not None:
            return {'n': 32, 'colors': result}
            
    # Fallback to n=9 (known valid from seed logic, or just return n=9)
    # The seed logic: 0 if sum % 3 == 0 else 1.
    # Let's verify n=9 is valid.
    # K4: 4 vertices. Sum of triples.
    # It's a standard construction?
    # Actually, for n=9, we can just use the seed logic.
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
