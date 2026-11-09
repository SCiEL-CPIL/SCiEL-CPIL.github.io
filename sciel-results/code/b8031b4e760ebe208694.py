"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a set of distinct integers in 1..256 such that:
    1. All nonempty subset sums are distinct.
    2. No two distinct subset sums divide one another.
    Maximizes the set size.
    """
    start_time = time.time()
    TIME_LIMIT = 55
    
    MAX_VAL = 256
    MAX_K = 30
    MAX_SUM = MAX_VAL * MAX_K
    
    present = [False] * (MAX_SUM + 1)
    current_sums = []
    best_set = []
    
    def backtrack(current_set, current_sums_list, present_arr):
        nonlocal best_set
        
        if time.time() - start_time > TIME_LIMIT:
            return
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        # Try adding each candidate x from 1 to MAX_VAL
        # To improve search, we can sort candidates. 
        # Smaller numbers are more likely to cause conflicts, so maybe try larger first?
        # Or smaller first to fill up? 
        # Let's try ascending order.
        
        for x in range(1, MAX_VAL + 1):
            if x in current_set:
                continue
            
            # Generate new sums
            new_sums_candidate = [x] + [s + x for s in current_sums_list]
            new_sums_candidate.sort()
            
            valid = True
            present_new = [False] * (MAX_SUM + 1)
            
            for v in new_sums_candidate:
                if v > MAX_SUM:
                    valid = False
                    break
                
                # Check if v is already in present_arr (existing sums)
                if present_arr[v]:
                    valid = False
                    break
                
                # Check if v is already in present_new (new sums added in this step)
                if present_new[v]:
                    valid = False
                    break
                
                # Check if any existing sum divides v
                # Iterate divisors of v
                # Optimization: if v is large, this is fast.
                for d in range(1, int(v**0.5) + 1):
                    if v % d == 0:
                        if present_arr[d] or present_new[d]:
                            valid = False
                            break
                        other = v // d
                        if other != d and (present_arr[other] or present_new[other]):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                
                # Check if v divides any existing sum
                # Iterate multiples of v
                for m in range(2 * v, MAX_SUM + 1, v):
                    if present_arr[m]:
                        valid = False
                        break
                if not valid:
                    break
                
                # If valid so far, mark v in present_new
                present_new[v] = True
            
            if valid:
                # Commit
                for v in new_sums_candidate:
                    present_arr[v] = True
                
                new_sums_list = current_sums_list + new_sums_candidate
                
                backtrack(current_set + [x], new_sums_list, present_arr)
                
                # Backtrack
                for v in new_sums_candidate:
                    present_arr[v] = False
        
    backtrack([], [], present)
    
    return list(best_set)

# EVOLVE-BLOCK-END
