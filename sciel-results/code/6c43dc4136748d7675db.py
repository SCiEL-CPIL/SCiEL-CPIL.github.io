import time

def construct():
    start_time = time.time()
    limit = 55.0
    
    current_set = []
    sums_set = set()
    
    # Iterate through all possible numbers in ascending order
    # This is a deterministic greedy approach
    for x in range(1, 4097):
        if time.time() - start_time > limit:
            break
            
        # Check if x can be added to current_set
        # We need to check all triples (a, b, c) from current_set U {x}
        # that include x.
        # The sums involving x are:
        # 1. 3x
        # 2. 2x + s for s in current_set
        # 3. x + s1 + s2 for s1, s2 in current_set (s1 <= s2)
        
        new_sums = []
        
        # 3x
        new_sums.append(3 * x)
        
        # 2x + s
        for s in current_set:
            new_sums.append(2 * x + s)
            
        # x + s1 + s2
        k = len(current_set)
        for i in range(k):
            s1 = current_set[i]
            for j in range(i, k):
                s2 = current_set[j]
                new_sums.append(x + s1 + s2)
                
        # Check for internal duplicates in new_sums
        # If any sum in new_sums is already in sums_set, or if new_sums has duplicates, x is invalid.
        
        # Optimization: Check against existing sums first
        valid = True
        for s_val in new_sums:
            if s_val in sums_set:
                valid = False
                break
                
        if not valid:
            continue
            
        # Check internal duplicates
        if len(new_sums) != len(set(new_sums)):
            continue
            
        # If valid, add x
        current_set.append(x)
        sums_set.update(new_sums)
        
    return current_set
