import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def score(U):
        if not U or 0 not in U:
            return 0.0
        U = sorted(set(U))
        n = len(U)
        if n < 2:
            return 0.0
        max_u = U[-1]
        if max_u == 0:
            return 0.0
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 0.0
        
        sums = set()
        diffs = set()
        for i in range(n):
            for j in range(n):
                sums.add(U[i] + U[j])
                diffs.add(U[i] - U[j])
        
        ratio = len(diffs) / len(sums)
        if ratio <= 1:
            return 1.0
        return 1.0 + math.log(ratio) / denom

    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Evaluate seed
    best_score = score(best_set)
    
    # Try powers of 2 minus 1
    for n in range(2, 12):
        if time.time() > deadline:
            break
        U = [2**i - 1 for i in range(n)]
        s = score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Local search to improve the best set found so far
    # We'll try adjusting individual elements to find better configurations
    current_set = sorted(best_set)
    current_score = best_score
    
    # Local search: try modifying each element slightly
    max_iterations = 500
    for iteration in range(max_iterations):
        if time.time() > deadline - 10.0:  # Keep safety margin
            break
        
        improved = False
        n = len(current_set)
        
        # Try each element
        for i in range(1, n):  # Keep 0 fixed
            original_val = current_set[i]
            
            # Try small perturbations
            for delta in [-3, -2, -1, 1, 2, 3]:
                new_val = original_val + delta
                if new_val <= 0 or new_val >= current_set[i+1] if i+1 < n else False:
                    continue
                
                # Create new set with modified element
                new_set = current_set.copy()
                new_set[i] = new_val
                new_set = sorted(set(new_set))
                
                if len(new_set) != n:
                    continue
                
                s = score(new_set)
                if s > current_score + 1e-12:
                    current_set = new_set
                    current_score = s
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # Try larger jumps occasionally
            if iteration % 10 == 5:
                for i in range(1, n):
                    original_val = current_set[i]
                    for delta in [-5, -4, 4, 5, -8, 8]:
                        new_val = original_val + delta
                        if new_val <= 0 or (i+1 < n and new_val >= current_set[i+1]):
                            continue
                        
                        new_set = current_set.copy()
                        new_set[i] = new_val
                        new_set = sorted(set(new_set))
                        
                        if len(new_set) != n:
                            continue
                        
                        s = score(new_set)
                        if s > current_score + 1e-12:
                            current_set = new_set
                            current_score = s
                            improved = True
                            break
                    if improved:
                        break
        
        if not improved and iteration > 50:
            break
    
    # Update best if local search improved
    if current_score > best_score:
        best_score = current_score
        best_set = current_set
    
    # Also try local search starting from other good seeds
    for base_n in [4, 5, 6]:
        if time.time() > deadline - 15.0:
            break
        
        base_set = [2**i - 1 for i in range(base_n)]
        base_score = score(base_set)
        
        if base_score < best_score - 0.01:
            continue
        
        # Local search from this base
        current_set = base_set.copy()
        current_score = base_score
        
        for iteration in range(200):
            if time.time() > deadline - 10.0:
                break
            
            improved = False
            n = len(current_set)
            
            for i in range(1, n):
                original_val = current_set[i]
                
                for delta in [-3, -2, -1, 1, 2, 3]:
                    new_val = original_val + delta
                    if new_val <= 0 or (i+1 < n and new_val >= current_set[i+1]):
                        continue
                    
                    new_set = current_set.copy()
                    new_set[i] = new_val
                    new_set = sorted(set(new_set))
                    
                    if len(new_set) != n:
                        continue
                    
                    s = score(new_set)
                    if s > current_score + 1e-12:
                        current_set = new_set
                        current_score = s
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_set = current_set

    return best_set
