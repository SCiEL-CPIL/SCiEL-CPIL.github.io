import time
import sys
import random

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    prescribed = [1, 2, 5, 11]
    initial_set = set(prescribed)
    
    def get_diffs(A):
        diffs = set()
        A_list = list(A)
        for i in range(len(A_list)):
            for j in range(i+1, len(A_list)):
                diffs.add(abs(A_list[i] - A_list[j]))
        return diffs

    initial_diffs = get_diffs(initial_set)
    
    best_set = initial_set.copy()
    best_size = len(best_set)
    
    def update_best(A):
        nonlocal best_set, best_size
        if len(A) > best_size:
            best_set = A.copy()
            best_size = len(best_set)
            # Optional: print or log, but not needed
            return True
        return False

    def backtrack(A, diffs, target_size, rng):
        nonlocal deadline
        
        if time.time() > deadline:
            return False
            
        update_best(A)
        
        if len(A) == target_size:
            return True
            
        A_list = list(A)
        candidates = []
        
        for x in range(1, 285):
            if x in A:
                continue
                
            new_diffs = set()
            valid = True
            for a in A_list:
                d = abs(x - a)
                if d in diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
                
            if valid:
                candidates.append((x, new_diffs))
                
        if not candidates:
            return False
            
        rng.shuffle(candidates)
        
        for x, new_diffs in candidates:
            A_new = A | {x}
            diffs_new = diffs | new_diffs
            
            if backtrack(A_new, diffs_new, target_size, rng):
                return True
                
        return False

    # Greedy warm-up
    def greedy_construct(rng):
        nonlocal deadline
        A = initial_set.copy()
        diffs = initial_diffs.copy()
        
        while time.time() < deadline:
            A_list = list(A)
            candidates = []
            for x in range(1, 285):
                if x in A:
                    continue
                new_diffs = set()
                valid = True
                for a in A_list:
                    d = abs(x - a)
                    if d in diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    candidates.append((x, new_diffs))
            
            if not candidates:
                break
                
            # Heuristic: Pick candidate that leaves most room? 
            # Or just random.
            x, new_diffs = rng.choice(candidates)
            A.add(x)
            diffs.update(new_diffs)
            update_best(A)
            
        return A

    # 1. Greedy pass to get a good starting point
    rng_greedy = random.Random(1)
    greedy_construct(rng_greedy)
    
    # 2. Backtracking with random restarts for target size 20
    # If we find 20, we return immediately.
    # If we don't, we return the best found (likely 18 or 19).
    
    restart = 0
    while time.time() < deadline:
        restart += 1
        rng = random.Random(restart)
        
        # We can try to extend from the current best_set?
        # Or always start from initial_set?
        # Starting from initial_set is safer for correctness of constraints, 
        # but starting from best_set might be faster if best_set is "good".
        # However, best_set might be a "dead end" for reaching 20.
        # Starting from initial_set explores the whole space.
        
        found = backtrack(initial_set, initial_diffs, 20, rng)
        
        if found:
            return list(best_set)
            
    return list(best_set)
