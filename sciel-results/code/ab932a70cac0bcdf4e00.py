import time
import random
import itertools

def _is_valid_incremental(S, x):
    """
    Check if adding x to S keeps the set valid.
    S is a list of ints.
    Returns True if valid.
    """
    # Compute existing sums
    sums = set()
    for i in range(1, len(S) + 1):
        for comb in itertools.combinations(S, i):
            sums.add(sum(comb))
    
    # New sums
    new_sums = set()
    new_sums.add(x)
    for s in sums:
        new_sums.add(s + x)
        
    # Check for duplicates
    if sums & new_sums:
        return False
        
    all_sums = sums | new_sums
    
    # Check divisibility
    # We only need to check pairs where at least one is in new_sums
    # But it's easier to just check all pairs in all_sums if size is small
    # Or optimize:
    # For each u in all_sums:
    #   if u in new_sums or ... 
    # Actually, if S was valid, sums are valid.
    # We need to check:
    # 1. u in new_sums, v in sums: u|v or v|u
    # 2. u in new_sums, v in new_sums: u|v or v|u (u!=v)
    
    # Efficient check:
    # Build a set of all sums for O(1) lookup
    all_sums_set = all_sums
    
    # Check new vs old
    for u in new_sums:
        # u divides v in sums?
        # v divides u?
        # Since u is new, we check all v in sums
        for v in sums:
            if u == v: continue # already checked duplicates
            if u < v:
                if v % u == 0:
                    return False
            else:
                if u % v == 0:
                    return False
                    
    # Check new vs new
    new_list = list(new_sums)
    for i in range(len(new_list)):
        u = new_list[i]
        for j in range(i+1, len(new_list)):
            v = new_list[j]
            if u < v:
                if v % u == 0:
                    return False
            else:
                if u % v == 0:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin
    
    best_set = []
    best_size = 0
    
    # Try multiple random orderings
    for _ in range(10000):
        if time.time() > deadline:
            break
            
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        current_set = []
        current_sums = set()
        
        for x in candidates:
            # Check if adding x is valid
            # Optimization: if time is running out, break
            if time.time() > deadline:
                break
            
            # Compute new sums
            new_sums = {x}
            for s in current_sums:
                new_sums.add(s + x)
                
            # Check duplicates
            if current_sums & new_sums:
                continue
                
            all_sums = current_sums | new_sums
            
            # Check divisibility
            valid = True
            # New vs Old
            for u in new_sums:
                for v in current_sums:
                    if u == v: continue
                    if u < v:
                        if v % u == 0:
                            valid = False
                            break
                    else:
                        if u % v == 0:
                            valid = False
                            break
                if not valid: break
            
            if valid:
                # New vs New
                new_list = list(new_sums)
                for i in range(len(new_list)):
                    u = new_list[i]
                    for j in range(i+1, len(new_list)):
                        v = new_list[j]
                        if u < v:
                            if v % u == 0:
                                valid = False
                                break
                        else:
                            if u % v == 0:
                                valid = False
                                break
                    if not valid: break
            
            if valid:
                current_set.append(x)
                current_sums = all_sums
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                    
                # If we reached a good size, we can continue to try to extend
                # or break if we think we found a local max?
                # Let's just continue.
        
        # End of restart
    
    return best_set
