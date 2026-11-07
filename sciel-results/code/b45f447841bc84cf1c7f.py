import time

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_set = []
    candidates = list(range(256, 0, -1))
    n_candidates = len(candidates)
    
    def is_valid_addition(sums, x):
        """
        sums: set of current subset sums
        x: new element
        Returns new set of sums if valid, else None.
        """
        new_sums_list = [x]
        for s in sums:
            new_sums_list.append(s + x)
        
        # Check intersection with old sums
        # If any new sum is in old sums, invalid.
        for ns in new_sums_list:
            if ns in sums:
                return None
        
        # Check divisibility
        # 1. New vs Old
        for ns in new_sums_list:
            for s in sums:
                # Check if one divides the other
                if ns > s:
                    if ns % s == 0:
                        return None
                else: # ns < s (ns != s already checked)
                    if s % ns == 0:
                        return None
        
        # 2. New vs New
        # We can sort new_sums_list to make it slightly easier?
        # Or just nested loop.
        # To optimize, we can note that if a | b, b = k*a.
        # Just checking modulo is fine.
        for i in range(len(new_sums_list)):
            ni = new_sums_list[i]
            for j in range(i + 1, len(new_sums_list)):
                nj = new_sums_list[j]
                if ni > nj:
                    if ni % nj == 0:
                        return None
                else:
                    if nj % ni == 0:
                        return None
        
        # Construct new set
        new_sums_set = set(new_sums_list)
        new_sums_set.update(sums)
        return new_sums_set

    def backtrack(current_set, current_sums, start_idx):
        nonlocal best_set
        
        if time.time() > deadline:
            return
            
        if len(current_set) > len(best_set):
            best_set = list(current_set)
            
        if start_idx >= n_candidates:
            return
            
        # Try adding candidates[start_idx]
        # We can also prune: if we don't add candidates[start_idx], we try start_idx+1
        # This is standard.
        
        # Optimization: If we have found a set of size K, and we are at depth D,
        # and remaining candidates < K - D, we can prune?
        # n_candidates - start_idx is the number of remaining candidates.
        # Max possible new size = len(current_set) + (n_candidates - start_idx)
        # If that is <= len(best_set), prune.
        if len(current_set) + (n_candidates - start_idx) <= len(best_set):
            return

        # Try including candidates[start_idx]
        x = candidates[start_idx]
        new_sums = is_valid_addition(current_sums, x)
        if new_sums is not None:
            backtrack(current_set + [x], new_sums, start_idx + 1)
            
        # Try excluding candidates[start_idx]
        backtrack(current_set, current_sums, start_idx + 1)

    backtrack([], set(), 0)
    
    return list(best_set)

# EVOLVE-BLOCK-END
