import time
from math import log
import itertools
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    # Use NumPy for faster computation
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums
    sums_arr = U_arr[:, None] + U_arr[None, :]
    unique_sums = np.unique(sums_arr)
    
    # Compute all pairwise differences
    diffs_arr = U_arr[:, None] - U_arr[None, :]
    unique_diffs = np.unique(diffs_arr)
    
    ratio = len(unique_diffs) / len(unique_sums)
    if ratio <= 1:
        return -1
        
    max_u = U_arr[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def construct():
    deadline = time.time() + 110  # Safety margin
    
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    # Update best if initial is good
    s = score_set(best_set)
    if s > best_score:
        best_score = s
        best_set = [0, 1, 4, 9, 11]
        
    # Phase 1: Exhaustive search for small sets
    # Increase max_val slightly and n range
    max_val = 50
    
    for n in range(3, 7): # Sizes 3 to 6
        if time.time() > deadline:
            break
            
        # If n is large, max_val should be smaller to keep combinations manageable
        # C(50, 5) is 2M, might be too slow with scoring.
        # C(40, 5) is 658k.
        # Let's cap max_val based on n
        current_max_val = min(max_val, 60 if n <= 4 else 40 if n <= 5 else 30)
        
        for combo in itertools.combinations(range(1, current_max_val + 1), n - 1):
            if time.time() > deadline:
                break
                
            U = (0,) + combo
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Phase 2: Local search with swaps
    if time.time() < deadline - 20:
        current_set = set(best_set)
        max_current = max(best_set)
        
        # Iterative improvement
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # 1. Try adding elements
            # Search range: 1 to max_current + 20
            for new_elem in range(1, max_current + 21):
                if time.time() > deadline:
                    break
                if new_elem in current_set:
                    continue
                new_set = current_set | {new_elem}
                s = score_set(list(new_set))
                if s > best_score:
                    best_score = s
                    best_set = list(new_set)
                    current_set = new_set
                    max_current = max(best_set)
                    improved = True
                    break # Restart loop to evaluate new state
            
            # 2. Try removing elements
            if improved:
                continue
            if len(current_set) > 2:
                for elem in list(current_set):
                    if time.time() > deadline:
                        break
                    if elem == 0:
                        continue
                    new_set = current_set - {elem}
                    if len(new_set) < 2:
                        continue
                    s = score_set(list(new_set))
                    if s > best_score:
                        best_score = s
                        best_set = list(new_set)
                        current_set = new_set
                        max_current = max(best_set)
                        improved = True
                        break # Restart loop
            
            # 3. Try swapping elements
            if improved:
                continue
            # Swap x in U with y not in U
            # Limit search space for y to keep it fast
            for x in list(current_set):
                if x == 0:
                    continue
                for y in range(1, max_current + 21):
                    if y in current_set:
                        continue
                    new_set = current_set - {x}
                    new_set.add(y)
                    s = score_set(list(new_set))
                    if s > best_score:
                        best_score = s
                        best_set = list(new_set)
                        current_set = new_set
                        max_current = max(best_set)
                        improved = True
                        break
                if improved:
                    break
    
    return best_set
