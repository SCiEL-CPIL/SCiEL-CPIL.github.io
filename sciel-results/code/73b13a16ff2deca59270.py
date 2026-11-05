import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 4096
    best_set = []
    best_size = 0
    
    def check_and_add(x, current_set, sums):
        new_sums = set()
        k = len(current_set)
        
        # (x, x, x)
        s = 3 * x
        if s in sums:
            return False, None
        new_sums.add(s)
        
        # (x, x, y)
        for y in current_set:
            s = 2 * x + y
            if s in sums or s in new_sums:
                return False, None
            new_sums.add(s)
            
        # (x, y, z)
        for i in range(k):
            y = current_set[i]
            # (x, y, y)
            s = x + 2 * y
            if s in sums or s in new_sums:
                return False, None
            new_sums.add(s)
            
            for j in range(i + 1, k):
                z = current_set[j]
                # (x, y, z)
                s = x + y + z
                if s in sums or s in new_sums:
                    return False, None
                new_sums.add(s)
                
        return True, new_sums

    def backtrack(start_val, current_set, sums):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        if start_val > n:
            return
            
        # Limit branching to 4 candidates per level
        candidates_tried = 0
        max_branches = 4
        
        for x in range(start_val, n + 1):
            if time.time() > deadline:
                return
                
            is_valid, new_sums = check_and_add(x, current_set, sums)
            
            if is_valid:
                current_set.append(x)
                new_total_sums = sums.union(new_sums)
                
                backtrack(x + 1, current_set, new_total_sums)
                
                current_set.pop()
                
                candidates_tried += 1
                if candidates_tried >= max_branches:
                    break
                    
    backtrack(1, [], set())
    
    # Fallback if no set found (unlikely)
    if not best_set:
        selected = []
        sums = set()
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            is_valid, new_sums = check_and_add(x, selected, sums)
            if is_valid:
                selected.append(x)
                sums = sums.union(new_sums)
        best_set = selected
        
    return best_set
