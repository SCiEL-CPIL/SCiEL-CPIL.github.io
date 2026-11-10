# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620  # 10s safety margin before 630s kill
    
    m = 205
    
    # Compute forbidden differences F = {x^2 mod 205 : x = 1..204} \ {0}
    forbidden = set()
    for x in range(1, m):
        r = (x * x) % m
        if r != 0:
            forbidden.add(r)
    
    # Precompute conflict matrix: conflict[i][j] = True if (i-j) mod m in forbidden
    # Since forbidden is symmetric (if d in F then (-d) mod m in F), we can use a set
    # conflict[i][j] = (i - j) % m in forbidden
    # For efficiency, precompute for each i the set of j that conflict with i
    conflicts = [set() for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i != j:
                d = (i - j) % m
                if d in forbidden:
                    conflicts[i].add(j)
    
    def is_valid_set(S):
        """Check if set S is valid (no forbidden differences)."""
        S_list = list(S)
        for idx in range(len(S_list)):
            for jdx in range(idx + 1, len(S_list)):
                d = (S_list[idx] - S_list[jdx]) % m
                if d in forbidden:
                    return False
        return True
    
    def greedy_build(order, start_set=None):
        """Build a valid set by greedily adding elements in given order."""
        if start_set is None:
            S = set()
        else:
            S = set(start_set)
        for elem in order:
            if elem in S:
                continue
            # Check if elem conflicts with any element in S
            ok = True
            for s in S:
                if elem in conflicts[s]:
                    ok = False
                    break
            if ok:
                S.add(elem)
        return S
    
    def local_search(S, max_iters=10000):
        """Try to improve set S via add/swap/remove operations."""
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try to add an element
            added = False
            for elem in range(m):
                if elem in S:
                    continue
                ok = True
                for s in S:
                    if elem in conflicts[s]:
                        ok = False
                        break
                if ok:
                    S.add(elem)
                    improved = True
                    added = True
                    break
            
            if added:
                continue
            
            # Try swap: remove one element, add another
            S_list = list(S)
            random.shuffle(S_list)
            for remove_elem in S_list:
                if time.time() >= deadline:
                    return S
                # Temporarily remove
                S_temp = set(S)
                S_temp.discard(remove_elem)
                # Try to add a new element
                candidates = [e for e in range(m) if e not in S_temp]
                random.shuffle(candidates)
                for add_elem in candidates:
                    ok = True
                    for s in S_temp:
                        if add_elem in conflicts[s]:
                            ok = False
                            break
                    if ok:
                        S = S_temp
                        S.add(add_elem)
                        improved = True
                        break
                if improved:
                    break
        
        return S
    
    best_set = set()
    
    # Multiple randomized greedy passes
    num_trials = 200
    for trial in range(num_trials):
        if time.time() >= deadline:
            break
        
        # Random ordering
        order = list(range(m))
        random.shuffle(order)
        
        # Greedy build
        S = greedy_build(order)
        
        # Local search to improve
        S = local_search(S, max_iters=5000)
        
        if len(S) > len(best_set):
            best_set = S
    
    # Additional local search on the best set with more iterations
    if time.time() < deadline:
        best_set = local_search(best_set, max_iters=20000)
    
    # Final verification
    if not is_valid_set(best_set):
        # Fallback: remove conflicting elements
        S_list = sorted(best_set)
        new_S = []
        for e in S_list:
            ok = True
            for f in new_S:
                if e in conflicts[f]:
                    ok = False
                    break
            if ok:
                new_S.append(e)
        best_set = set(new_S)
    
    # Ensure we have at least one element
    if len(best_set) == 0:
        best_set = {0}
    
    return np.array(sorted(best_set), dtype=np.int64)

# EVOLVE-BLOCK-END
