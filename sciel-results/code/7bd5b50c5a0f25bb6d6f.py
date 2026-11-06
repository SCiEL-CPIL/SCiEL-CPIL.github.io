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
        k4_to_triples = []
        for k4 in combinations(range(n), 4):
            ts = [triple_to_idx[t] for t in combinations(k4, 3)]
            k4s.append(ts)
            
        k5s = []
        for k5 in combinations(range(n), 5):
            ts = [triple_to_idx[t] for t in combinations(k5, 3)]
            k5s.append(ts)
            
        # Map triple index to affected subgraphs
        # affected[t] = list of (type, idx)
        # type 0 for K4, 1 for K5
        affected = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k4s):
            for t in ts:
                affected[t].append((0, i))
        for i, ts in enumerate(k5s):
            for t in ts:
                affected[t].append((1, i))
                
        num_k4 = len(k4s)
        num_k5 = len(k5s)
        
        # State
        colors = np.random.randint(0, 2, num_triples)
        # violations: 0 for K4, 1 for K5? No, separate arrays or one big array.
        # Let's use one array `viol` of size num_k4 + num_k5.
        # viol[0:num_k4] for K4s, viol[num_k4:num_k4+num_k5] for K5s.
        viol = np.zeros(num_k4 + num_k5, dtype=np.int8)
        
        def calc_violation_k4(idx):
            ts = k4s[idx]
            c = colors[ts[0]]
            # Check if all are c
            # Using numpy sum is faster for small arrays?
            # sum(colors[ts] == c) == 4
            return 1 if np.all(colors[ts] == c) else 0
            
        def calc_violation_k5(idx):
            ts = k5s[idx]
            c = colors[ts[0]]
            return 1 if np.all(colors[ts] == c) else 0
            
        # Initialize violations
        for i in range(num_k4):
            viol[i] = calc_violation_k4(i)
        for i in range(num_k5):
            viol[num_k4 + i] = calc_violation_k5(i)
            
        E = np.sum(viol)
        
        if E == 0:
            return colors.tolist()
            
        # Local Search
        # Simulated Annealing
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        
        steps = 0
        max_steps = 5000000 # Large number, time limited
        
        while time.time() < start_time + time_limit:
            if T < min_T:
                break
                
            t_idx = random.randint(0, num_triples - 1)
            old_c = colors[t_idx]
            new_c = 1 - old_c
            
            # Calculate delta E
            delta = 0
            # We need to check affected subgraphs
            # For each affected subgraph, check old state and new state
            # To avoid modifying colors until we accept, we can just check logic.
            # Or we can modify, calc, revert if rejected.
            # Modifying and reverting is safer and simpler with numpy.
            
            colors[t_idx] = new_c
            
            # Recalculate viol for affected subgraphs
            # We need to store old viol values to compute delta?
            # Or just recalculate E? No, E is sum.
            # Let's just recalculate the viol values for affected subgraphs and update E.
            
            # We need to know the old viol values.
            # Let's store them.
            old_viols = {}
            for typ, idx in affected[t_idx]:
                if typ == 0:
                    old_viols[(typ, idx)] = viol[idx]
                else:
                    old_viols[(typ, idx)] = viol[num_k4 + idx]
                    
            # Update viol
            for typ, idx in affected[t_idx]:
                if typ == 0:
                    viol[idx] = calc_violation_k4(idx)
                else:
                    viol[num_k4 + idx] = calc_violation_k5(idx)
                    
            # Calculate delta
            for (typ, idx), old_v in old_viols.items():
                if typ == 0:
                    new_v = viol[idx]
                else:
                    new_v = viol[num_k4 + idx]
                delta += new_v - old_v
                
            if delta <= 0:
                E += delta
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < np.exp(-delta / T):
                    E += delta
                else:
                    # Revert
                    colors[t_idx] = old_c
                    for (typ, idx), old_v in old_viols.items():
                        if typ == 0:
                            viol[idx] = old_v
                        else:
                            viol[num_k4 + idx] = old_v
                            
            T *= cooling
            steps += 1
            
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
