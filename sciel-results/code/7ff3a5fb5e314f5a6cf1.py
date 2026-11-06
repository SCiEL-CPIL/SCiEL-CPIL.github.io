import random
import time

def construct():
    mandatory = {1, 2, 5, 11}
    max_val = 284
    target_size = 20
    global_start = time.time()
    time_limit = 115.0
    
    def get_sums(S):
        sums = set()
        L = sorted(S)
        for i in range(len(L)):
            for j in range(i, len(L)):
                sums.add(L[i] + L[j])
        return sums

    def is_compatible(x, S, sums):
        # Check if x can be added to S
        # Sums in S are distinct.
        # New sums will be {x + y for y in S} U {2*x}
        # These must not intersect with existing sums.
        # Also, the new sums themselves must be distinct?
        # If x + y1 == x + y2, then y1 == y2, which is impossible since y are distinct in S.
        # If x + y1 == 2*x, then y1 == x, which is impossible since x not in S.
        # So we just need to check if any new sum is in `sums`.
        for y in S:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def add_to_sums(x, S, sums):
        new_sums = set(sums)
        for y in S:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        return new_sums

    # Initial state
    current_set = list(mandatory)
    current_sums = get_sums(current_set)
    
    # Candidates are all numbers in 1..284 not in mandatory
    candidates = [x for x in range(1, max_val + 1) if x not in mandatory]
    
    best_set = list(current_set)
    best_size = len(current_set)

    # DFS function
    def dfs(idx, S, sums):
        nonlocal best_set, best_size, global_start, time_limit
        
        if time.time() - global_start > time_limit:
            return False
            
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            
        if len(S) == target_size:
            return True
            
        # Heuristic: We need to pick (target_size - len(S)) elements from candidates[idx:]
        # If not enough candidates left, fail
        remaining_needed = target_size - len(S)
        if len(candidates) - idx < remaining_needed:
            return False
            
        # Try to pick next element
        # We can shuffle the slice of candidates? No, that breaks the "idx" logic for uniqueness.
        # Instead, we can randomize the order of candidates ONCE at the start of each DFS call?
        # No, that would allow picking the same element twice if we're not careful.
        # The standard way: candidates is a list. We pick from candidates[idx:].
        # To randomize, we should shuffle the entire candidates list BEFORE starting a DFS run.
        
        for i in range(idx, len(candidates)):
            x = candidates[i]
            
            if is_compatible(x, S, sums):
                new_S = S + [x]
                new_sums = add_to_sums(x, S, sums)
                
                if dfs(i + 1, new_S, new_sums):
                    return True
                    
        return False

    # Strategy: Multiple random restarts of the candidate order
    # Each restart, we shuffle candidates and run DFS.
    # DFS is deterministic given the order.
    
    while time.time() - global_start < time_limit:
        random.shuffle(candidates)
        
        # We run DFS. It will update best_set if it finds larger sets.
        # If it finds size 20, it returns True.
        found = dfs(0, current_set, current_sums)
        
        if found:
            break
            
        # If not found, we might want to try a slightly different heuristic?
        # Or just keep shuffling.
        
    return sorted(best_set)
