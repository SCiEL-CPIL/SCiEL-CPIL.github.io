import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    N = 128
    base_set = set(range(65, 129))
    candidates = list(range(1, 65))
    
    def time_left():
        return deadline - time.time()
    
    def is_forbidden_triple_with_new(a, b, c, S, new_elem):
        """
        Check if {a, b, c} is a forbidden triple in S U {new_elem}.
        a, b, c are distinct.
        """
        s_ab = a + b
        s_ac = a + c
        s_bc = b + c
        
        # Sums must be <= N to be in the universe
        if s_ab > N or s_ac > N or s_bc > N:
            return False
            
        # Check membership in S or if sum equals new_elem
        if not ((s_ab in S) or (s_ab == new_elem)):
            return False
        if not ((s_ac in S) or (s_ac == new_elem)):
            return False
        if not ((s_bc in S) or (s_bc == new_elem)):
            return False
            
        return True

    def can_add(S, x):
        """
        Check if x can be added to S without creating a forbidden triple.
        """
        elems = list(S)
        k = len(elems)
        for i in range(k):
            a = elems[i]
            for j in range(i + 1, k):
                b = elems[j]
                if is_forbidden_triple_with_new(a, b, x, S, x):
                    return False
        return True

    best_size = len(base_set)
    best_set = set(base_set)
    
    rng = random.Random(42)
    
    # Multiple restarts with different random orders
    while time_left() > 5.0:
        # Greedy phase
        current_set = set(base_set)
        
        # Randomize candidate order
        cand_order = candidates.copy()
        rng.shuffle(cand_order)
        
        for x in cand_order:
            if time_left() < 5.0:
                break
            if can_add(current_set, x):
                current_set.add(x)
        
        # Local Search Phase: Try to improve by swaps or additions
        improved = True
        while improved and time_left() > 5.0:
            improved = False
            
            # 1. Try to add any remaining candidate
            # Iterate in random order to break ties
            add_cand = candidates.copy()
            rng.shuffle(add_cand)
            for x in add_cand:
                if x in current_set:
                    continue
                if can_add(current_set, x):
                    current_set.add(x)
                    improved = True
                    # Restart the loop to check for more additions
                    break 
            
            if improved:
                continue
                
            # 2. Try swaps: Remove one from S, add one from not-S
            if current_set:
                remove_candidates = list(current_set)
                rng.shuffle(remove_candidates)
                
                for r in remove_candidates:
                    if time_left() < 5.0:
                        break
                    temp_set = set(current_set)
                    temp_set.remove(r)
                    
                    # Try to add a random element not in temp_set
                    # Focus on candidates 1..64 first, then maybe others if we removed some?
                    # Base set is 65..128. We only added from 1..64.
                    # So elements not in temp_set are from 1..64 (not added) OR 65..128 (if we removed one).
                    # But we start with all 65..128. We only remove from current_set.
                    # So add_candidates should be all 1..128 not in temp_set.
                    add_candidates = [x for x in range(1, N + 1) if x not in temp_set]
                    rng.shuffle(add_candidates)
                    
                    for a in add_candidates:
                        if can_add(temp_set, a):
                            new_set = set(temp_set)
                            new_set.add(a)
                            
                            # After swap, try to fill up again
                            fill_cand = candidates.copy()
                            rng.shuffle(fill_cand)
                            for x in fill_cand:
                                if x in new_set:
                                    continue
                                if can_add(new_set, x):
                                    new_set.add(x)
                            
                            if len(new_set) > len(current_set):
                                current_set = new_set
                                improved = True
                                break
                    if improved:
                        break
        
        # Update best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    # Convert to sorted list of ints
    result = sorted(list(best_set))
    return result
