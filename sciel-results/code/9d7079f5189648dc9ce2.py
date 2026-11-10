import time

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    time_limit = 85.0
    
    candidates = list(range(256, 0, -1))
    best_set = []
    
    # sums_by_card[k] is a set of integers
    # min_max[k] is (min_sum, max_sum) for cardinality k
    
    def is_valid_addition(x, sums_by_card, min_max, current_size):
        # Generate new sums and check collisions
        # We can do this in one pass
        
        # We need to check new sums of size k against old sums of size j != k
        
        # To avoid generating all new sums if they are obviously disjoint,
        # we can check bounds first.
        
        # But we need the actual sums to check intersection if bounds overlap.
        
        # Let's generate new_sums for each k, but check bounds before intersection.
        
        new_sums_list = [set() for _ in range(current_size + 2)]
        
        for k in range(1, current_size + 2):
            if k-1 < len(sums_by_card):
                prev_sums = sums_by_card[k-1]
                if not prev_sums:
                    continue
                
                # Calculate min/max of new sums
                # min_new = min(prev_sums) + x
                # max_new = max(prev_sums) + x
                # We have min_max[k-1]
                
                min_prev, max_prev = min_max[k-1]
                min_new = min_prev + x
                max_new = max_prev + x
                
                # Check if this range overlaps with any j != k
                # If it doesn't overlap with ANY j != k, we still need to generate sums?
                # No, if it doesn't overlap, it's valid w.r.t to those j.
                # But we need to add them to new_sums_by_card for future steps.
                # So we must generate them.
                
                # However, we can skip the intersection check if bounds are disjoint.
                
                # Let's generate the set first
                new_sums = set()
                for s in prev_sums:
                    new_sums.add(s + x)
                new_sums_list[k] = new_sums
                
                # Now check intersection with j != k
                # Optimization: Only check j where ranges might overlap
                for j in range(0, current_size + 1):
                    if j == k:
                        continue
                    if j >= len(min_max):
                        continue
                    
                    min_j, max_j = min_max[j]
                    if max_new < min_j or min_new > max_j:
                        continue
                    
                    # Ranges overlap, check actual intersection
                    if new_sums & sums_by_card[j]:
                        return False
        
        return True

    def update_sums(sums_by_card, min_max, x, current_size):
        new_sums_by_card = [set(s) for s in sums_by_card]
        if len(new_sums_by_card) < current_size + 2:
             new_sums_by_card.extend([set() for _ in range(current_size + 2 - len(new_sums_by_card))])
        
        new_min_max = list(min_max)
        if len(new_min_max) < current_size + 2:
             new_min_max.extend([(0,0) for _ in range(current_size + 2 - len(new_min_max))])
             
        for k in range(1, current_size + 2):
            prev_sums = sums_by_card[k-1] if k-1 < len(sums_by_card) else set()
            if not prev_sums:
                continue
                
            new_sums = set()
            min_new = float('inf')
            max_new = float('-inf')
            
            for s in prev_sums:
                ns = s + x
                new_sums.add(ns)
                if ns < min_new: min_new = ns
                if ns > max_new: max_new = ns
            
            # Merge with existing new_sums_by_card[k]
            existing = new_sums_by_card[k]
            existing.update(new_sums)
            
            # Update min/max
            if k < len(min_max):
                old_min, old_max = min_max[k]
                # If existing was empty before, min_max[k] might be 0,0?
                # We should track if set is empty.
                # Actually, if set is empty, we shouldn't have min_max.
                # Let's assume min_max[k] is valid if sums_by_card[k] is not empty.
                
                # If existing was empty, min_max[k] was dummy.
                # If new_sums is empty, min_max[k] should be dummy.
                
                if existing:
                    # Combine old and new
                    # But old min_max is for old set. New set is old U new_sums.
                    # So new_min = min(old_min, min_new)
                    # new_max = max(old_max, max_new)
                    # But only if old set was not empty.
                    # How do we know if old set was empty?
                    # We can check if sums_by_card[k] was empty.
                    
                    if sums_by_card[k]:
                        old_min, old_max = min_max[k]
                        new_min_max[k] = (min(old_min, min_new), max(old_max, max_new))
                    else:
                        new_min_max[k] = (min_new, max_new)
                else:
                    new_min_max[k] = (0, 0) # Dummy for empty
            
        return new_sums_by_card, new_min_max

    def backtrack(idx, current_set, sums_by_card, min_max, current_size):
        nonlocal best_set
        
        if time.time() - start_time > time_limit:
            return
            
        if current_size > len(best_set):
            best_set = list(current_set)
            
        remaining = len(candidates) - idx
        if current_size + remaining <= len(best_set):
            return

        for i in range(idx, len(candidates)):
            x = candidates[i]
            
            if is_valid_addition(x, sums_by_card, min_max, current_size):
                new_sums, new_mins = update_sums(sums_by_card, min_max, x, current_size)
                current_set.append(x)
                backtrack(i + 1, current_set, new_sums, new_mins, current_size + 1)
                current_set.pop()
                
    initial_sums = [{0}]
    initial_min_max = [(0, 0)]
    initial_set = []
    
    backtrack(0, initial_set, initial_sums, initial_min_max, 0)
    
    return list(best_set)

# EVOLVE-BLOCK-END
