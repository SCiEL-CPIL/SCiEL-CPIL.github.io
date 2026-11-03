import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = 0
    best_set = [0, 1, 3]
    
    def evaluate(U):
        if not U: return 0
        U_arr = np.array(U)
        max_u = np.max(U_arr)
        if max_u == 0:
            return 1.0
        
        # Use sets for small N, numpy for larger?
        # Numpy is generally faster for unique on large arrays
        sums = np.add.outer(U_arr, U_arr).flatten()
        diffs = np.subtract.outer(U_arr, U_arr).flatten()
        
        size_sum = len(np.unique(sums))
        size_diff = len(np.unique(diffs))
        
        if size_sum == 0: return 0
        ratio = size_diff / size_sum
        if ratio <= 0: return 0
        
        log_ratio = np.log(ratio)
        log_max = np.log(2 * max_u + 1)
        
        if log_max == 0: return 1.0
        
        return 1 + log_ratio / log_max

    # Heuristic 1: Random sets
    for N in [3, 4, 5, 10, 20, 50]:
        for _ in range(10):
            # Generate random set with 0
            # Scale to keep max small initially
            max_val = N * N 
            U = set([0])
            while len(U) < N:
                U.add(random.randint(0, max_val))
            U = list(U)
            s = evaluate(U)
            if s > best_score:
                best_score = s
                best_set = U
                
    # Heuristic 2: Base expansion
    for B in [2, 3, 4, 5]:
        for k in range(1, 10):
            U = [0]
            val = 0
            for i in range(k):
                # Add B^i
                U.append(val + (1 if i==0 else 0)) # This is messy
            # Better: generate all sums of distinct powers
            powers = [B**i for i in range(k)]
            # Generate all subsets
            # This is 2^k, too big for k>15
            if k <= 12:
                U_set = set([0])
                for p in powers:
                    new_els = [x + p for x in U_set]
                    U_set.update(new_els)
                U = list(U_set)
                s = evaluate(U)
                if s > best_score:
                    best_score = s
                    best_set = U

    # Local Search
    # Start from best_set
    current_set = set(best_set)
    current_score = best_score
    
    # Define neighborhood
    # 1. Swap an element
    # 2. Add an element
    # 3. Remove an element
    
    while time.time() < deadline:
        # Pick a move
        move_type = random.choice(['swap', 'add', 'remove'])
        
        new_set = set(current_set)
        
        if move_type == 'swap' and len(new_set) > 1:
            u = random.choice(list(new_set))
            # New value
            # Scale based on current max
            max_u = max(new_set) if new_set else 1
            u_new = random.randint(0, max_u + 10)
            new_set.remove(u)
            new_set.add(u_new)
            
        elif move_type == 'add' and len(new_set) < 100: # Cap size
            max_u = max(new_set) if new_set else 1
            u_new = random.randint(0, max_u + 10)
            new_set.add(u_new)
            
        elif move_type == 'remove' and len(new_set) > 1:
            u = random.choice(list(new_set))
            if u == 0: continue # Must contain 0
            new_set.remove(u)
        else:
            continue
            
        if 0 not in new_set:
            new_set.add(0)
            
        s = evaluate(list(new_set))
        
        if s > current_score:
            current_set = new_set
            current_score = s
            if s > best_score:
                best_score = s
                best_set = list(current_set)
        else:
            # Simulated annealing? Or just reject?
            # For now, reject to keep it simple and fast.
            # Or accept with small prob to escape local minima.
            pass

    return best_set
