import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    best_set = []
    best_size = 0
    
    def check_validity_incremental(current_sums, x):
        """
        Checks if adding x to a set with current_sums is valid.
        current_sums: set of all subset sums of the current set (including 0)
        Returns (True, new_sums) if valid, (False, None) otherwise.
        """
        # Generate new sums: x + s for s in current_sums
        new_sums = set()
        for s in current_sums:
            new_sums.add(s + x)
        
        # Check distinctness: new_sums and current_sums must be disjoint
        # (excluding that 0 is in current_sums but not in new_sums since x > 0)
        if new_sums & current_sums:
            return (False, None)
        
        # Combined sums
        all_sums = current_sums | new_sums
        max_sum = max(all_sums) if all_sums else 0
        
        # Check divisibility: no two distinct sums in all_sums (excluding 0) divide each other
        # We only need to check pairs where at least one is in new_sums (excluding 0)
        # since current_sums (excluding 0) is already known to be an antichain
        
        non_empty_new = {s for s in new_sums if s != 0}
        non_empty_old = {s for s in current_sums if s != 0}
        
        # Internal check for new_sums
        new_list = list(non_empty_new)
        for i in range(len(new_list)):
            u = new_list[i]
            for j in range(i + 1, len(new_list)):
                v = new_list[j]
                if v > u:
                    if v % u == 0:
                        return (False, None)
                else:
                    if u % v == 0:
                        return (False, None)
        
        # Cross check between old and new
        for u in non_empty_new:
            for v in non_empty_old:
                if v > u:
                    if v % u == 0:
                        return (False, None)
                else:
                    if u % v == 0:
                        return (False, None)
        
        return (True, all_sums)
    
    # Strategy: multiple random restarts with different orderings
    # For each restart, try to build a set greedily
    
    # First, try the deterministic approach (256 down to 1)
    current_sums = {0}
    current_set = []
    for x in range(256, 0, -1):
        if time.time() > deadline:
            break
        valid, new_sums = check_validity_incremental(current_sums, x)
        if valid:
            current_sums = new_sums
            current_set.append(x)
    
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = list(current_set)
    
    # Now try random restarts
    random.seed(42)
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Create a random ordering of candidates
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        current_sums = {0}
        current_set = []
        
        for x in candidates:
            if time.time() > deadline:
                break
            valid, new_sums = check_validity_incremental(current_sums, x)
            if valid:
                current_sums = new_sums
                current_set.append(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
            # If we found a good set, try to extend it with remaining candidates
            remaining = [x for x in range(1, 257) if x not in current_set]
            for x in remaining:
                if time.time() > deadline:
                    break
                valid, new_sums = check_validity_incremental(current_sums, x)
                if valid:
                    current_sums = new_sums
                    current_set.append(x)
                    best_size = len(current_set)
                    best_set = list(current_set)
    
    # Return as list of ints
    return list(best_set)
