import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    initial_set = [1, 2, 5, 11]
    initial_sums = set()
    for i in range(len(initial_set)):
        for j in range(i, len(initial_set)):
            initial_sums.add(initial_set[i] + initial_set[j])
            
    best_set = initial_set[:]
    best_size = len(initial_set)
    
    # We want to find a set of size 20
    target_size = 20
    
    def is_valid_addition(x, current_set, used_sums):
        # Check if x can be added to current_set
        # Sums: x + y for y in current_set, and 2*x
        # Must not be in used_sums
        # And must be distinct among themselves (x+y1 != x+y2 if y1!=y2, which is true since y1!=y2)
        # So just check against used_sums
        for y in current_set:
            if x + y in used_sums:
                return False
        if 2 * x in used_sums:
            return False
        return True

    def get_new_sums(x, current_set):
        new_sums = set()
        for y in current_set:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        return new_sums

    def search(current_set, used_sums, candidates):
        nonlocal best_set, best_size, deadline, start_time
        
        if time.time() > deadline:
            return False
            
        if len(current_set) > best_size:
            best_set = current_set[:]
            best_size = len(current_set)
            
        if len(current_set) == target_size:
            return True
            
        if not candidates:
            return False
            
        # Pruning: if remaining candidates < needed elements
        needed = target_size - len(current_set)
        if len(candidates) < needed:
            return False

        for i, x in enumerate(candidates):
            if time.time() > deadline:
                return False
                
            if not is_valid_addition(x, current_set, used_sums):
                continue
                
            # Add x
            new_sums = get_new_sums(x, current_set)
            # Optimization: check if new_sums has any intersection with used_sums
            # is_valid_addition already did this, but let's be safe or just trust it.
            # Actually, is_valid_addition checks x+y and 2x.
            
            new_set = current_set + [x]
            new_used_sums = used_sums | new_sums
            
            # Filter candidates for next step
            # Candidates must be > x
            # And must be valid with new_set
            next_candidates = []
            for y in candidates[i+1:]:
                if is_valid_addition(y, new_set, new_used_sums):
                    next_candidates.append(y)
                    
            if search(new_set, new_used_sums, next_candidates):
                return True
                
    # Initial candidates: 12 to 284
    # We can pre-filter candidates that are invalid with the initial set
    initial_candidates = []
    for x in range(12, 285):
        if is_valid_addition(x, initial_set, initial_sums):
            initial_candidates.append(x)
            
    search(initial_set, initial_sums, initial_candidates)
    
    return best_set
