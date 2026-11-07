import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Candidates: 256 down to 2. 
    # 1 is excluded because if 1 is in S, 1 is a subset sum, and 1 divides everything else, 
    # so no other subset sum can exist. Thus size is max 1. 
    # We want size > 1, so exclude 1.
    candidates = list(range(256, 1, -1))
    n = len(candidates)
    
    best_set = []
    best_sums = set()
    
    def is_valid_addition(existing_sums, new_sums):
        # Check if any sum in new_sums divides or is divisible by any in existing_sums
        # Also check within new_sums
        all_sums = list(existing_sums) + list(new_sums)
        # Optimization: We only need to check new_sums against existing_sums 
        # and new_sums against new_sums.
        # Existing sums were already pairwise non-dividing.
        
        # 1. Check new_sums against existing_sums
        for ns in new_sums:
            for es in existing_sums:
                if ns == es:
                    return False # Should not happen if sums are unique, but safe
                if ns < es:
                    if es % ns == 0:
                        return False
                else:
                    if ns % es == 0:
                        return False
        
        # 2. Check new_sums against each other
        # Since new_sums are generated from a single new element x, 
        # they are {x} U {s+x | s in existing}.
        # We need to ensure no two in new_sums divide each other.
        new_list = list(new_sums)
        for i in range(len(new_list)):
            for j in range(i+1, len(new_list)):
                a = new_list[i]
                b = new_list[j]
                if a == b:
                    return False
                if a < b:
                    if b % a == 0:
                        return False
                else:
                    if a % b == 0:
                        return False
        return True

    def search(idx, current_set, current_sums):
        nonlocal best_set, best_sums
        
        if time.time() > deadline:
            return
        
        # Update best
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            best_sums = current_sums.copy()
            
        # Pruning: Can we beat the best?
        if len(current_set) + (n - idx) <= len(best_set):
            return
            
        if idx == n:
            return
        
        # Option 1: Include candidates[idx]
        x = candidates[idx]
        # New sums are {x} U {s + x for s in current_sums}
        new_sums_set = {x}
        for s in current_sums:
            new_sums_set.add(s + x)
            
        if is_valid_addition(current_sums, new_sums_set):
            new_set = current_set + [x]
            new_sums = current_sums.union(new_sums_set)
            search(idx + 1, new_set, new_sums)
            
        # Option 2: Exclude candidates[idx]
        search(idx + 1, current_set, current_sums)

    search(0, [], set())
    
    # Ensure return type is list of ints
    return best_set

# EVOLVE-BLOCK-START
# The construct function is defined above, but the block structure requires it inside or referencing it.
# The prompt says "Keep the function names and signatures of the seed program".
# The seed program had construct() inside the block. I will place the whole logic inside the block.
# EVOLVE-BLOCK-END
