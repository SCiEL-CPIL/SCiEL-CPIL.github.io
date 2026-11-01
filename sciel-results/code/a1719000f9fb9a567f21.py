import time
import math
import random
import numpy as np

def _score_of_fast(U):
    """
    Optimized score calculation.
    """
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(int(x) for x in U))
    n = len(U)
    if n < 2:
        return 0.0
    
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    
    sumset = set()
    diffset = set()
    
    add_s = sumset.add
    add_d = diffset.add
    
    for i in range(n):
        ui = U[i]
        for j in range(n):
            add_s(ui + U[j])
            add_d(ui - U[j])
            
    size_sum = len(sumset)
    size_diff = len(diffset)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    denom_arg = 2 * max_u + 1
    if denom_arg <= 1:
        return 0.0
        
    denom = math.log(denom_arg)
    if denom == 0:
        return 0.0
        
    return 1 + math.log(ratio) / denom

def _local_search_structured(U, time_limit=10.0, allow_change_size=True):
    """
    Perform local search with structured moves.
    Moves:
    1. Shift an element (small step).
    2. Add a new element (strategic placement).
    3. Remove an element.
    4. 'Block shift' perturbation.
    """
    best_U = sorted(set(int(x) for x in U))
    if 0 not in best_U:
        best_U.insert(0, 0)
    best_U.sort()
    best_score = _score_of_fast(best_U)
    
    start_time = time.time()
    n = len(best_U)
    
    for step in range(5000):
        if time.time() - start_time > time_limit:
            break
            
        # Choose a move type
        r = random.random()
        
        if r < 0.6:
            # Shift a random element
            if n <= 1:
                continue
            i = random.randint(1, n - 1)
            
            # Determine range for new value
            low = best_U[i-1] + 1
            # Allow expansion if it's the last element
            if i == n - 1:
                high = best_U[i] + random.randint(1, 20)
            else:
                high = best_U[i+1] - 1
                if low > high:
                    continue
            
            new_val = random.randint(low, high)
            
            new_U = best_U.copy()
            new_U[i] = new_val
            new_U.sort()
            
            # Check uniqueness
            if len(new_U) != n:
                continue
                
        elif r < 0.8 and allow_change_size:
            # Add a new element
            if n > 25: # Cap size to keep search space manageable
                continue
            max_val = best_U[-1]
            
            # Strategy: Add a value that creates new differences.
            # Pick a random position in the range [1, max_val * 2]
            # Bias towards values that are "far" from existing ones
            # Or try adding near gaps
            if random.random() < 0.5:
                # Add near the end
                new_val = max_val + random.randint(1, 10)
            else:
                # Add in a gap
                if n > 2:
                    idx = random.randint(1, n-1)
                    gap_low = best_U[idx-1] + 1
                    gap_high = best_U[idx] - 1
                    if gap_low < gap_high:
                        new_val = random.randint(gap_low, gap_high)
                    else:
                        new_val = random.randint(1, max_val)
                else:
                    new_val = random.randint(1, max_val + 10)
            
            if new_val in best_U:
                continue
                
            new_U = best_U + [new_val]
            new_U.sort()
            
        elif r < 0.9 and allow_change_size:
            # Remove an element
            if n <= 2:
                continue
            # Don't remove 0
            i = random.randint(1, n - 1)
            new_U = best_U[:i] + best_U[i+1:]
            
        else:
            # Two-block perturbation
            # Try to modify the set to look like A union B where A is small and dense, B is large and sparse
            if n < 4:
                continue
                
            # Pick a pivot
            pivot_idx = random.randint(1, n-1)
            # Try to shift the upper part by a large amount
            shift = random.randint(10, 100)
            new_U = best_U.copy()
            for k in range(pivot_idx, n):
                new_U[k] += shift
            new_U.sort()
            
            if len(new_U) != n:
                continue

        new_score = _score_of_fast(new_U)
        
        if new_score > best_score:
            best_score = new_score
            best_U = new_U
            n = len(best_U)
        else:
            # Simulated annealing style acceptance to escape local optima
            # Accept with probability p if worse
            if new_score > best_score - 0.001:
                if random.random() < 0.05:
                    best_U = new_U
                    best_score = new_score
                    n = len(best_U)

    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_global_U = [0, 1, 3]
    best_global_score = _score_of_fast(best_global_U)
    
    # Known optimal Golomb rulers
    golomb_rulers = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 4, 10, 12, 17, 25, 34],
        [0, 1, 4, 10, 12, 17, 25, 34, 42],
        [0, 1, 4, 10, 12, 17, 25, 34, 42, 52],
        [0, 1, 4, 10, 12, 17, 25, 34, 42, 52, 60],
        [0, 1, 4, 10, 12, 17, 25, 34, 42, 52, 60, 72],
    ]
    
    # Evaluate initial candidates
    for U in golomb_rulers:
        if time.time() > deadline:
            break
        s = _score_of_fast(U)
        if s > best_global_score:
            best_global_score = s
            best_global_U = U
    
    # Local search on each Golomb ruler
    candidates_for_search = []
    for U in golomb_rulers:
        s = _score_of_fast(U)
        candidates_for_search.append((s, U))
    
    # Sort by score descending
    candidates_for_search.sort(key=lambda x: x[0], reverse=True)
    
    # Search the top 5 candidates
    for score, U in candidates_for_search[:5]:
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        t_limit = min(10.0, remaining / 3) # Divide remaining time among searches
        
        U_opt, s_opt = _local_search_structured(U, time_limit=t_limit)
        
        if s_opt > best_global_score:
            best_global_score = s_opt
            best_global_U = U_opt

    # Try some random restarts with different structures
    # 1. Random Golomb-like
    # 2. Two-block structures
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        if remaining < 10:
            break
            
        # Random initial set
        n = random.randint(4, 12)
        U_rand = [0]
        curr = 0
        for _ in range(n - 1):
            curr += random.randint(1, 15)
            U_rand.append(curr)
            
        # Or try a two-block structure
        if random.random() < 0.5:
            n1 = random.randint(2, 5)
            n2 = random.randint(2, 5)
            M = random.randint(20, 100)
            A = list(range(n1))
            B = [M + i for i in range(n2)]
            U_rand = A + B
            U_rand.sort()
            
        t_limit = min(5.0, remaining / 4)
        U_opt, s_opt = _local_search_structured(U_rand, time_limit=t_limit)
        
        if s_opt > best_global_score:
            best_global_score = s_opt
            best_global_U = U_opt
            
    return best_global_U
