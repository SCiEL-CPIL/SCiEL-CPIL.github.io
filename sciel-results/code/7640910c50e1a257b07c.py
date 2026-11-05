import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    prescribed = [1, 2, 5, 11]
    max_val = 284
    
    # Pre-compute sums for prescribed
    base_sums = set()
    for i in range(len(prescribed)):
        for j in range(i, len(prescribed)):
            base_sums.add(prescribed[i] + prescribed[j])
    
    best_set = prescribed[:]
    
    def is_valid_addition(x, current_set, current_sums):
        # Check if x creates duplicate sums
        # New sums: x + y for y in current_set, and 2*x
        for y in current_set:
            s = x + y
            if s in current_sums:
                return False
        if 2 * x in current_sums:
            return False
        # Check if new sums collide with each other
        # x+y1 = x+y2 => y1=y2 (distinct in set)
        # x+y = 2x => y=x (x not in set)
        # So internal collisions are impossible if x not in set.
        return True

    def search(current_set, current_sums, target_size, start_val):
        nonlocal best_set, deadline
        
        if time.time() > deadline:
            return None
        
        if len(current_set) == target_size:
            return current_set[:]
        
        # Try to add elements
        # We can iterate from start_val to max_val
        # Heuristic: try values that are "far" from existing ones? 
        # Or just linear?
        # Linear is simple.
        
        for x in range(start_val, max_val + 1):
            if time.time() > deadline:
                return None
                
            if x in current_set:
                continue
                
            if is_valid_addition(x, current_set, current_sums):
                new_set = current_set + [x]
                # Update sums
                new_sums = set(current_sums)
                for y in current_set:
                    new_sums.add(x + y)
                new_sums.add(2 * x)
                
                result = search(new_set, new_sums, target_size, x + 1)
                if result is not None:
                    return result
                    
        return None

    # Try to find max size
    # Max possible is 20
    for size in range(20, len(prescribed), -1):
        if time.time() > deadline:
            break
            
        # Sort prescribed to start search from a reasonable point
        # The search function expects sorted set for efficiency? 
        # My `search` appends to the end, so `current_set` stays sorted if we iterate x increasing.
        # Prescribed is [1, 2, 5, 11].
        
        res = search(prescribed[:], base_sums, size, 12)
        if res is not None:
            best_set = res
            break
            
    return best_set
