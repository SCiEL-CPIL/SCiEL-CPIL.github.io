import time
import random
import numpy as np
import math

def calculate_score(U):
    if not U:
        return 1.0
    u = np.array(sorted(list(U)), dtype=np.int64)
    n = len(u)
    if n == 0:
        return 1.0
    max_u = u[-1]
    if max_u == 0:
        return 1.0
    
    # Compute sums and differences
    # Using outer product
    sums = np.add.outer(u, u)
    diffs = np.subtract.outer(u, u)
    
    # Get unique counts
    # np.unique flattens and sorts, then removes duplicates
    unique_sums = np.unique(sums)
    unique_diffs = np.unique(diffs)
    
    count_sums = len(unique_sums)
    count_diffs = len(unique_diffs)
    
    if count_sums == 0:
        return 1.0
        
    ratio = count_diffs / count_sums
    if ratio <= 1.0:
        return 1.0
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Keep 10s buffer
    
    best_set = {0, 1, 3}
    best_score = calculate_score(best_set)
    
    # 1. Exhaustive Search for small M
    # Try M=16 (subsets of 1..16)
    max_m_exhaustive = 16
    # Iterate through all non-empty subsets of {1, ..., max_m_exhaustive}
    # Add 0 to each
    for mask in range(1, 1 << max_m_exhaustive):
        if time.time() > deadline:
            break
        U = {0}
        for i in range(max_m_exhaustive):
            if mask & (1 << i):
                U.add(i + 1)
        s = calculate_score(U)
        if s > best_score:
            best_score = s
            best_set = U

    # 2. Local Search (Hill Climbing)
    # Start from best found
    current_set = set(best_set)
    current_score = best_score
    max_val = 500 # Search range for new elements
    
    # Add some random restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        if restart == 0:
            current_set = set(best_set)
            current_score = best_score
        else:
            # Random initialization
            size = random.randint(3, 20)
            current_set = {0}
            while len(current_set) < size:
                current_set.add(random.randint(1, max_val))
            current_score = calculate_score(current_set)
            
        # Hill Climb
        stuck_count = 0
        while time.time() < deadline:
            improved = False
            
            # Try adding an element
            if len(current_set) < 50:
                candidate = set(current_set)
                candidate.add(random.randint(1, max_val))
                s = calculate_score(candidate)
                if s > current_score:
                    current_set = candidate
                    current_score = s
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_set = set(current_set)
            
            # Try removing an element
            if len(current_set) > 2:
                candidate = set(current_set)
                elems = [x for x in candidate if x != 0]
                if elems:
                    candidate.remove(random.choice(elems))
                    s = calculate_score(candidate)
                    if s > current_score:
                        current_set = candidate
                        current_score = s
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_set = set(current_set)
                        
            # Try swapping
            if len(current_set) > 1:
                candidate = set(current_set)
                elems = [x for x in candidate if x != 0]
                if elems:
                    old = random.choice(elems)
                    new = random.randint(1, max_val)
                    if new not in candidate:
                        candidate.remove(old)
                        candidate.add(new)
                        s = calculate_score(candidate)
                        if s > current_score:
                            current_set = candidate
                            current_score = s
                            improved = True
                            if s > best_score:
                                best_score = s
                                best_set = set(current_set)
            
            if not improved:
                stuck_count += 1
                if stuck_count > 100:
                    break # Break out of hill climb for this restart
            else:
                stuck_count = 0
                
    return sorted(list(best_set))
