"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Powers of 4 form a valid base set with no triple-sum collisions
    base = [4**i for i in range(10) if 4**i <= 4096]
    # 1, 4, 16, 64, 256, 1024, 4096
    
    def can_add(lst, candidate):
        """Check if adding candidate to lst keeps all triple sums unique."""
        n = len(lst)
        if n == 0:
            return True
        
        arr = np.array(lst, dtype=np.int64)
        cand = np.int64(candidate)
        
        # Compute all existing triple sums using broadcasting
        # For n elements, triples (i,j,k) with i<=j<=k
        # We'll generate all valid triples
        idx = np.triu_indices(n)
        # For each pair (i,j), k ranges from j to n-1
        # This is complex to vectorize directly, so let's use a different approach
        
        # Alternative: compute all pair sums, then for each element add to pair sums
        # Pair sums: lst[i] + lst[j] for i <= j
        pair_sums = set()
        for i in range(n):
            for j in range(i, n):
                pair_sums.add(lst[i] + lst[j])
        
        # Triple sums = for each k, lst[k] + (lst[i] + lst[j]) for i <= j
        # But we need to avoid duplicates in the triple generation
        # Actually, let me just use the O(n^2) approach more carefully
        
        # Existing triple sums
        existing_sums = set()
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    existing_sums.add(lst[i] + lst[j] + lst[k])
        
        # New triples with one candidate: cand + lst[i] + lst[j] for i <= j
        for i in range(n):
            for j in range(i, n):
                s = candidate + lst[i] + lst[j]
                if s in existing_sums:
                    return False
                existing_sums.add(s)
        
        # New triples with two candidates: 2*cand + lst[i]
        for i in range(n):
            s = 2 * candidate + lst[i]
            if s in existing_sums:
                return False
            existing_sums.add(s)
        
        # New triple with three candidates: 3*cand
        s = 3 * candidate
        if s in existing_sums:
            return False
        
        return True
    
    def can_add_fast(lst, candidate, existing_triple_sums):
        """Faster check using precomputed triple sums."""
        n = len(lst)
        
        # Check new triples against existing
        # One candidate: cand + lst[i] + lst[j] for i <= j
        for i in range(n):
            for j in range(i, n):
                s = candidate + lst[i] + lst[j]
                if s in existing_triple_sums:
                    return False
        
        # Two candidates: 2*cand + lst[i]
        for i in range(n):
            s = 2 * candidate + lst[i]
            if s in existing_triple_sums:
                return False
        
        # Three candidates: 3*cand
        s = 3 * candidate
        if s in existing_triple_sums:
            return False
        
        return True
    
    def compute_triple_sums(lst):
        """Compute all triple sums for a list."""
        n = len(lst)
        sums = set()
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums.add(lst[i] + lst[j] + lst[k])
        return sums
    
    def try_greedy_insert(base_set, candidates_order, deadline, max_size=80):
        """Try to insert candidates in given order."""
        current = list(base_set)
        current_sums = compute_triple_sums(current)
        
        for cand in candidates_order:
            if time.time() > deadline:
                break
            if len(current) >= max_size:
                break
            
            # Check if cand can be added
            n = len(current)
            valid = True
            
            # One candidate triples
            for i in range(n):
                for j in range(i, n):
                    s = cand + current[i] + current[j]
                    if s in current_sums:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Two candidate triples
                for i in range(n):
                    s = 2 * cand + current[i]
                    if s in current_sums:
                        valid = False
                        break
            
            if valid:
                s = 3 * cand
                if s in current_sums:
                    valid = False
            
            if valid:
                current.append(cand)
                # Update triple sums
                n_old = n
                for i in range(n_old):
                    for j in range(i, n_old):
                        current_sums.add(cand + current[i] + current[j])
                for i in range(n_old):
                    current_sums.add(2 * cand + current[i])
                current_sums.add(3 * cand)
        
        return current
    
    best_set = list(base)
    
    # Multiple random restarts with different candidate orders
    all_candidates = [x for x in range(1, 4097) if x not in set(base)]
    
    num_attempts = 0
    while time.time() < deadline - 10:
        num_attempts += 1
        # Use different random seeds for diversity
        random.seed(num_attempts * 7919)
        shuffled = list(all_candidates)
        random.shuffle(shuffled)
        
        result = try_greedy_insert(base, shuffled, deadline)
        
        if len(result) > len(best_set):
            best_set = result
            if len(best_set) >= 80:
                break
    
    # Local search: try to swap elements to improve cardinality
    if len(best_set) < 80 and time.time() < deadline - 5:
        current = list(best_set)
        current_sums = compute_triple_sums(current)
        
        # Try to remove one element and add two new ones
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for remove_idx in range(len(current)):
                if time.time() > deadline - 2:
                    break
                
                temp = current[:remove_idx] + current[remove_idx+1:]
                temp_sums = compute_triple_sums(temp)
                
                # Try to add candidates
                added = 0
                for cand in random.sample(all_candidates, min(500, len(all_candidates))):
                    if time.time() > deadline - 2:
                        break
                    if len(temp) + added >= 80:
                        break
                    
                    n = len(temp)
                    valid = True
                    for i in range(n):
                        for j in range(i, n):
                            s = cand + temp[i] + temp[j]
                            if s in temp_sums:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        for i in range(n):
                            s = 2 * cand + temp[i]
                            if s in temp_sums:
                                valid = False
                                break
                    
                    if valid:
                        s = 3 * cand
                        if s in temp_sums:
                            valid = False
                    
                    if valid:
                        temp.append(cand)
                        n_old = n
                        for i in range(n_old):
                            for j in range(i, n_old):
                                temp_sums.add(cand + temp[i] + temp[j])
                        for i in range(n_old):
                            temp_sums.add(2 * cand + temp[i])
                        temp_sums.add(3 * cand)
                        added += 1
                
                if added > 1:  # Net gain
                    current = temp
                    current_sums = temp_sums
                    improved = True
                    if len(current) >= 80:
                        break
                    if len(current) > len(best_set):
                        best_set = list(current)
    
    return best_set[:80]
# EVOLVE-BLOCK-END
