import time
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0
    N = 4096
    MAX_SIZE = 80
    
    def check_and_add(x, selected, triple_sums):
        k = len(selected)
        if k == 0:
            triple_sums.add(3 * x)
            return [x]
        
        sel = np.array(selected, dtype=np.int64)
        
        total_new = 1 + k + k * (k + 1) // 2
        
        new_sums = np.empty(total_new, dtype=np.int64)
        new_sums[0] = 3 * x
        new_sums[1:1+k] = 2 * x + sel
        
        if k > 1:
            yz = sel[:, None] + sel[None, :]
            mask = np.triu(np.ones((k, k), dtype=bool))
            yz_upper = yz[mask]
            new_sums[1+k:1+k+len(yz_upper)] = x + yz_upper
        elif k == 1:
            new_sums[1+k] = x + sel[0] + sel[0]
        
        new_list = new_sums.tolist()
        if len(new_list) != len(set(new_list)):
            return None
        
        new_set = set(new_list)
        if new_set & triple_sums:
            return None
        
        triple_sums.update(new_set)
        return selected + [x]
        
    def try_greedy(offset=0, step=1, start=1):
        selected = []
        triple_sums = set()
        
        # Initialize with a few elements if offset > 0?
        # Let's just start the loop from 'start'
        
        for x in range(start, N + 1, step):
            if len(selected) >= MAX_SIZE:
                break
            if time.time() > DEADLINE - 10:
                break
            
            res = check_and_add(x, selected, triple_sums)
            if res is not None:
                selected = res
        return selected

    best_set = []
    
    # Strategy 1: Standard Greedy
    s1 = try_greedy()
    if len(s1) > len(best_set):
        best_set = s1
        
    # Strategy 2: Greedy with a small skip at the beginning (to see if skipping 1 helps)
    # Sometimes 1 is too "tight".
    s2 = try_greedy(start=2)
    if len(s2) > len(best_set):
        best_set = s2
        # If we started at 2, we might be able to add 1 back?
        # Try to add 1 to s2
        # (Simplified: just try to insert 1)
        
    # Strategy 3: Local Search / Swap on best_set
    # Try to replace the largest element with a smaller one to "compress"
    # Then try to add more elements.
    
    if best_set:
        selected = list(best_set)
        triple_sums = set()
        # Rebuild triple_sums for the current set
        sel_arr = np.array(selected, dtype=np.int64)
        # This is O(k^3) to rebuild, but k is small.
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                for l in range(j, len(selected)):
                    triple_sums.add(selected[i] + selected[j] + selected[l])
                    
        improved = True
        while improved and time.time() < DEADLINE - 5:
            improved = False
            # Try to swap largest element with a smaller one
            if selected:
                largest = selected[-1]
                # Find smallest possible replacement
                # We want to replace 'largest' with 't' < largest
                # Try t from 1 upwards
                for t in range(1, largest):
                    if t in selected:
                        continue
                    # Check if replacing largest with t is valid
                    # Remove largest's contributions, add t's contributions
                    
                    # This is expensive to check properly without removing from triple_sums first.
                    # Let's do a full check on the new set.
                    new_selected = selected[:-1] + [t]
                    new_selected.sort()
                    
                    # Check validity of new_selected
                    valid = True
                    new_sums = set()
                    ns = np.array(new_selected, dtype=np.int64)
                    k = len(ns)
                    # Generate all sums
                    for i in range(k):
                        for j in range(i, k):
                            for l in range(j, k):
                                s_val = ns[i] + ns[j] + ns[l]
                                if s_val in new_sums:
                                    valid = False
                                    break
                                new_sums.add(s_val)
                            if not valid: break
                        if not valid: break
                    if valid:
                        # Check against existing triple_sums? 
                        # No, we are replacing, so we need to check if new_sums are unique 
                        # AND if they conflict with the sums of the REMAINING elements?
                        # Actually, if we just check the full set validity, it's sufficient.
                        # But we also need to ensure we didn't break the "distinctness" 
                        # within the new set. The loop above does that.
                        # However, we also need to ensure the new set is distinct from... 
                        # Wait, the condition is "Every unordered triple... must have a different sum".
                        # This applies to the set itself. 
                        # So if new_selected is a valid B3 set, it's fine.
                        
                        # But we also want to make sure we can ADD more elements later.
                        # Compressing might help.
                        
                        # Let's accept the swap
                        selected = new_selected
                        # Rebuild triple_sums for the new set
                        triple_sums = new_sums
                        improved = True
                        break # Restart the search for the next swap/add
                
                # Try to add a new element
                if not improved and time.time() < DEADLINE - 5:
                    # Try to add smallest available x
                    for x in range(1, N + 1):
                        if x in selected:
                            continue
                        res = check_and_add(x, selected, triple_sums)
                        if res is not None:
                            selected = res
                            improved = True
                            break
                            
    return best_set if len(best_set) > len(selected) else selected
