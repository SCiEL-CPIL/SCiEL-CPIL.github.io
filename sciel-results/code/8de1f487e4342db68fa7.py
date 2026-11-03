import numpy as np
import itertools
import random
import time

def score_set(U):
    U = list(set(U))
    if 0 not in U:
        return -1
    U.sort()
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Calculate sumset
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    
    sum_size = len(sumset)
    diff_size = len(diffset)
    
    if sum_size == 0:
        return -1
        
    ratio = diff_size / sum_size
    if ratio <= 0:
        return -1
        
    log_ratio = np.log2(ratio)
    log_max = np.log2(2 * max_u + 1)
    
    if log_max == 0:
        return -1
        
    return 1 + log_ratio / log_max

def construct():
    random.seed(42)
    best_score = -1
    best_set = [0, 1, 3, 7]
    
    # Initial candidates
    candidates = []
    
    # 1. Small exhaustive search (from original, but faster)
    # We keep the original exhaustive search for small sets as a baseline
    # but we will also run heuristic search.
    
    # 2. Heuristic Search
    # We will try to find a set with size n in [5, 20] and max value M in [50, 2000]
    
    def local_search(initial_set, max_iter=200):
        current_set = list(set(initial_set))
        if 0 not in current_set:
            return current_set, -1
        current_score = score_set(current_set)
        
        for _ in range(max_iter):
            improved = False
            # Try modifying each element
            n = len(current_set)
            # Sort to maintain consistency, but we work on list
            # We need to be careful with duplicates in the list representation if we just swap
            # Let's maintain a sorted list of unique elements
            
            # Candidate generation:
            # 1. Replace an element
            # 2. Add an element
            # 3. Remove an element
            
            # To speed up, we limit the search space for new values
            max_val = max(current_set) if current_set else 10
            min_val = min(current_set) if current_set else 0
            
            # Neighborhood size
            neighborhood = 5
            
            # Try replacing
            for i in range(n):
                old_val = current_set[i]
                # Try values around old_val
                for delta in range(-neighborhood, neighborhood + 1):
                    new_val = old_val + delta
                    if new_val < 0:
                        continue
                    if new_val == 0 and i != 0 and 0 in current_set:
                         # 0 is already there, if we replace a non-zero with 0, we get duplicate
                         # We must ensure uniqueness
                         pass 
                    
                    new_set = current_set.copy()
                    # Remove old_val (careful if duplicates exist in list? We keep unique)
                    # Since we keep unique, just replace
                    new_set[i] = new_val
                    # Check uniqueness
                    if len(set(new_set)) < len(new_set):
                        continue
                    if 0 not in new_set:
                        continue
                    
                    s = score_set(new_set)
                    if s > current_score:
                        current_score = s
                        current_set = new_set
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding
            if not improved and n < 15:
                # Try adding a value near existing ones or random
                for _ in range(10):
                    # Pick a random base
                    base_idx = random.randint(0, n-1)
                    base_val = current_set[base_idx]
                    new_val = base_val + random.randint(-10, 10)
                    if new_val < 0:
                        continue
                    if new_val in current_set:
                        continue
                    new_set = current_set + [new_val]
                    s = score_set(new_set)
                    if s > current_score:
                        current_score = s
                        current_set = new_set
                        improved = True
                        break
            
            # Try removing
            if not improved and n > 3:
                for i in range(n):
                    if current_set[i] == 0:
                        continue # Don't remove 0
                    new_set = current_set.copy()
                    new_set.pop(i)
                    s = score_set(new_set)
                    if s > current_score:
                        current_score = s
                        current_set = new_set
                        improved = True
                        break
            
            if not improved:
                break
                
        return current_set, current_score

    # Run multiple random starts
    for trial in range(50):
        # Random size
        n = random.randint(5, 12)
        # Random max value
        max_val = random.randint(50, 500)
        
        # Generate random set containing 0
        # Ensure 0 is in the set
        if random.random() < 0.5:
             # Pick n-1 random distinct values from 1..max_val
             vals = random.sample(range(1, max_val+1), n-1)
             U_init = [0] + vals
        else:
             # Pick n random distinct values from 0..max_val, ensure 0 is in
             vals = random.sample(range(0, max_val+1), n)
             if 0 not in vals:
                 vals[0] = 0 # Force 0? No, just replace one
                 # Better: sample n-1 from 1..max_val and add 0
                 vals = [0] + random.sample(range(1, max_val+1), n-1)
             U_init = vals
        
        # Run local search
        found_set, found_score = local_search(U_init, max_iter=100)
        
        if found_score > best_score:
            best_score = found_score
            best_set = found_set

    # Also try some structured sets
    # Geometric
    for base in [2, 3, 4]:
        U_geo = [base**i for i in range(8)]
        if 0 not in U_geo: U_geo = [0] + U_geo
        found_set, found_score = local_search(U_geo, max_iter=100)
        if found_score > best_score:
            best_score = found_score
            best_set = found_set

    # Arithmetic
    for step in [1, 2, 3]:
        U_ar = list(range(0, 20, step))
        found_set, found_score = local_search(U_ar, max_iter=100)
        if found_score > best_score:
            best_score = found_score
            best_set = found_set

    return best_set
