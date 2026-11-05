import time
import random
import numpy as np

def set_sizes(A):
    A = list(A)
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return len(sums), len(prods)

def objective(A):
    s, p = set_sizes(A)
    return max(s, p)

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin under 90s
    
    # Helper to validate
    def is_valid(A):
        if len(A) != 20: return False
        if len(set(A)) != 20: return False
        return all(1 <= x <= 10000 for x in A)

    best_set = None
    best_obj = float('inf')
    
    def update_best(A):
        nonlocal best_set, best_obj
        if not is_valid(A):
            return
        obj = objective(A)
        if obj < best_obj:
            best_obj = obj
            best_set = A[:]

    # 1. Generate Seeds
    seeds = []
    
    # Seed 1: AP
    seeds.append(list(range(1, 21)))
    
    # Seed 2: GP Base 2
    seeds.append([2**i for i in range(20)])
    
    # Seed 3: GP Base 1.5
    gp = [int(round(1.5 ** i)) for i in range(20)]
    seen = set()
    uniq = []
    for x in gp:
        if x not in seen and 1 <= x <= 10000:
            seen.add(x)
            uniq.append(x)
    while len(uniq) < 20:
        uniq.append(uniq[-1] + 1)
    seeds.append(uniq[:20])
    
    # Seed 4: Greedy Construction
    # Start with 1, then add elements that minimize the objective
    A_greedy = [1]
    # Pre-compute some candidate pools to speed up greedy step
    # We will search a window around the current max and multiples
    for _ in range(19):
        current_max = max(A_greedy)
        # Candidate generation
        cands = set()
        # Add neighbors of existing elements
        for a in A_greedy:
            for d in range(-2, 3):
                if 1 <= a + d <= 10000:
                    cands.add(a + d)
            # Multiples
            for m in [2, 3]:
                if a * m <= 10000:
                    cands.add(a * m)
        # Add some randoms in the range
        for _ in range(50):
            cands.add(random.randint(1, min(10000, current_max * 2 + 10)))
        
        best_cand = None
        best_cand_obj = float('inf')
        
        # To make it faster, we don't evaluate all cands if the set is large, 
        # but n=20 is small.
        for c in cands:
            if c in A_greedy:
                continue
            A_temp = A_greedy + [c]
            obj = objective(A_temp)
            if obj < best_cand_obj:
                best_cand_obj = obj
                best_cand = c
        
        if best_cand is not None:
            A_greedy.append(best_cand)
        else:
            # Fallback
            A_greedy.append(current_max + 1)
            
    seeds.append(A_greedy)
    
    # Seed 5: Random Geometric
    for r in [1.2, 1.3, 1.4, 1.6, 1.8, 2.0]:
        gp = [int(round(r ** i)) for i in range(20)]
        seen = set()
        uniq = []
        for x in gp:
            if x not in seen and 1 <= x <= 10000:
                seen.add(x)
                uniq.append(x)
        while len(uniq) < 20:
            uniq.append(uniq[-1] + 1)
        seeds.append(uniq[:20])

    # Evaluate seeds
    for s in seeds:
        update_best(s)

    # 2. Local Search / Simulated Annealing
    # We will run SA from the best seed and a few others
    
    def simulated_annealing(A_init, time_limit):
        nonlocal best_set, best_obj
        A = list(A_init)
        cur_obj = objective(A)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_set = A[:]
            
        T = 10.0
        cooling = 0.99
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            if T < 0.01:
                # Restart from best
                A = best_set[:]
                cur_obj = best_obj
                T = 10.0
                
            # Propose move
            i = random.randint(0, 19)
            old_val = A[i]
            
            # Generate candidate
            # Mix of local and global
            if random.random() < 0.7:
                # Local move
                # Try values near old_val
                candidates = set()
                for d in range(-10, 11):
                    candidates.add(old_val + d)
                # Multiplicative neighbors
                for f in [0.5, 0.8, 1.2, 1.5, 2.0]:
                    candidates.add(int(old_val * f))
                candidates.discard(old_val)
                valid_cands = [c for c in candidates if 1 <= c <= 10000 and c not in A]
                if not valid_cands:
                    continue
                new_val = random.choice(valid_cands)
            else:
                # Global move
                new_val = random.randint(1, 10000)
                if new_val in A:
                    continue
            
            A[i] = new_val
            new_obj = objective(A)
            
            # Metropolis acceptance
            if new_obj < cur_obj or random.random() < np.exp((cur_obj - new_obj) / T):
                cur_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_set = A[:]
            else:
                A[i] = old_val
                
            T *= cooling

    # Run SA on the best seed
    if time.time() < deadline - 10:
        simulated_annealing(best_set, deadline - time.time() - 10)
        
    # Run SA on other good seeds if time permits
    # Sort seeds by objective
    evaluated_seeds = []
    for s in seeds:
        if is_valid(s):
            evaluated_seeds.append((objective(s), s))
    evaluated_seeds.sort(key=lambda x: x[0])
    
    for obj, s in evaluated_seeds[1:4]: # Top 3 others
        if time.time() > deadline - 5:
            break
        simulated_annealing(s, deadline - time.time() - 5)
        
    # Final polish
    if time.time() < deadline - 2:
        simulated_annealing(best_set, deadline - time.time() - 2)

    return best_set
