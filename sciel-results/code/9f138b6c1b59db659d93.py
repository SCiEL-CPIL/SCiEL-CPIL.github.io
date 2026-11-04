import math
import random
import time

def score_set(U):
    U = sorted(set(U))
    if not U or U[0] != 0:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute |U+U|
    sum_set = set()
    for i in range(len(U)):
        for j in range(i, len(U)):
            sum_set.add(U[i] + U[j])
    
    # Compute |U-U|
    diff_set = set()
    for i in range(len(U)):
        for j in range(len(U)):
            diff_set.add(U[i] - U[j])
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0
        
    ratio = size_diff / size_sum
    
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def construct():
    start_time = time.time()
    timeout = 110.0  # Leave 10s margin
    
    best_set = [0, 1, 4, 6]
    best_score = score_set(best_set)
    
    # Try various sizes N
    for N in range(3, 15):
        # Initial range for values
        # Heuristic: max value roughly N^2 for good Golomb-like sets
        max_val = 2 * N * N + 10
        
        # Multiple random restarts
        for restart in range(20):
            if time.time() - start_time > timeout:
                break
                
            # Initialize a random set
            # Ensure 0 is included
            U = {0}
            # Pick N-1 distinct random integers from 1 to max_val
            while len(U) < N:
                U.add(random.randint(1, max_val))
            
            U = list(U)
            current_score = score_set(U)
            
            # Local search / Hill Climbing
            improved = True
            while improved and (time.time() - start_time < timeout):
                improved = False
                # Generate neighbors
                # 1. Replace an element
                for i in range(len(U)):
                    if time.time() - start_time > timeout:
                        break
                    original_val = U[i]
                    # Try a few random replacements
                    for _ in range(5):
                        new_val = random.randint(0, max_val)
                        if new_val in U and U.index(new_val) != i:
                            continue
                        U[i] = new_val
                        if 0 not in U:
                            U[0] = 0 # Keep 0? Or just ensure 0 is in set.
                            # Better: just ensure 0 is always in the set during moves?
                            # If we replace 0, we lose it.
                            # Let's fix index 0 to be 0?
                            # No, U is a list. Let's just make sure 0 is in the set.
                            if 0 not in U:
                                # Replace a random non-zero element with 0?
                                # Or just revert?
                                # Simplest: don't allow 0 to be replaced.
                                # But U is a list, indices change.
                                # Let's just check if 0 is in U. If not, skip this move.
                                U[i] = original_val
                                continue
                        s = score_set(U)
                        if s > current_score:
                            current_score = s
                            improved = True
                        else:
                            U[i] = original_val
                
                # 2. Swap two elements? Not very useful if values are distinct.
                # 3. Add/remove element? N is fixed in this loop.
                
            if current_score > best_score:
                best_score = current_score
                best_set = U[:]
                
    return best_set
