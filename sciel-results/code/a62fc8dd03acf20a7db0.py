<<<<<<< SEARCH
import time
import sys

def _can_add(existing_sums, current_set, x):
    """
    Check if x can be added to current_set without creating duplicate triple sums.
    Returns a list of new sums if valid, else None.
    """
    new_sums = []
    n = len(current_set)
    
    # (x, x, x)
    s_3x = 3 * x
    if s_3x in existing_sums:
        return None
    new_sums.append(s_3x)
    
    # (x, x, y) and (x, y, y) for y in current_set
    for y in current_set:
        s_2xy = 2 * x + y
        if s_2xy in existing_sums:
            return None
        new_sums.append(s_2xy)
        
        s_xy2 = x + 2 * y
        if s_xy2 in existing_sums:
            return None
        new_sums.append(s_xy2)
            
    # (x, y, z) for y < z in current_set
    for i in range(n):
        y = current_set[i]
        for j in range(i + 1, n):
            z = current_set[j]
            s_xyz = x + y + z
            if s_xyz in existing_sums:
                return None
            new_sums.append(s_xyz)
            
    return new_sums

def _compute_all_sums(current_set):
    """Compute all triple sums for a given set."""
    sums = set()
    n = len(current_set)
    for i in range(n):
        a = current_set[i]
        for j in range(i, n):
            b = current_set[j]
            for k in range(j, n):
                c = current_set[k]
                sums.add(a + b + c)
    return sums

def _greedy_build(n_max, start=1, end=None):
    """
    Build a B_3[1] set using a greedy strategy.
    """
    if end is None:
        end = n_max
        
    current_set = []
    existing_sums = set()
    
    for x in range(start, end + 1):
        new_sums = _can_add(existing_sums, current_set, x)
        if new_sums is not None:
            current_set.append(x)
            existing_sums.update(new_sums)
            
    return current_set

def _try_increase_size(current_set, n_max, time_limit):
    """
    Try to increase the size of the set by 1 by removing one element 
    and adding two others.
    """
    start_time = time.time()
    n_curr = len(current_set)
    
    # Sort to ensure deterministic behavior and easier removal logic
    current_set = sorted(current_set)
    
    # Try removing each element
    for i in range(n_curr):
        if time.time() - start_time > time_limit:
            break
            
        # Remove element i
        removed_val = current_set[i]
        temp_set = current_set[:i] + current_set[i+1:]
        
        # We need to add 2 elements to temp_set to get back to n_curr + 1
        # Let's try to find the first valid 'a' and then the first valid 'b'
        
        # Compute sums for temp_set
        temp_sums = _compute_all_sums(temp_set)
        
        found_a = False
        # Try adding 'a'
        for a in range(1, n_max + 1):
            if a == removed_val:
                continue
            
            new_sums_a = _can_add(temp_sums, temp_set, a)
            if new_sums_a is None:
                continue
                
            # 'a' is valid. Now try to find 'b'
            # Update temp_set and sums with 'a'
            set_with_a = temp_set + [a]
            set_with_a.sort()
            sums_with_a = temp_sums.union(new_sums_a)
            
            # Try adding 'b' > a (to avoid duplicates and maintain order logic)
            for b in range(a + 1, n_max + 1):
                if b == removed_val:
                    continue
                    
                new_sums_b = _can_add(sums_with_a, set_with_a, b)
                if new_sums_b is not None:
                    # We found a valid increase!
                    new_set = set_with_a + [b]
                    return sorted(new_set)
            
            # If we got here, 'a' was valid but no 'b' was found for this 'a'
            # We can break the 'a' loop if we decide to try the next 'removed_val'
            # But maybe a larger 'a' works? 
            # However, searching all a for all i is expensive.
            # Let's break after finding a valid 'a' that fails to find 'b'?
            # No, different 'a' might allow different 'b'.
            # But to keep it fast, let's limit the search for 'b' or 'a'.
            # If we don't find b quickly, this 'i' might not be fruitful.
            break # Move to next removed element? Or continue searching a?
            # Let's continue searching a, but with a time check.
            
    return current_set # No improvement found

def construct():
    start_time = time.time()
    deadline = start_time + 85 # 5s safety margin
    
    n_max = 4096
    
    # Initial greedy build
    best_set = _greedy_build(n_max)
    best_size = len(best_set)
    
    # Local search loop
    improvement_found = True
    while improvement_found:
        improvement_found = False
        
        # Try to improve the current best set
        # We give a small time slice per attempt
        time_slice = min(1.0, (deadline - time.time()) / 10)
        if time_slice <= 0:
            break
            
        improved_set = _try_increase_size(best_set, n_max, time_slice)
        
        if len(improved_set) > len(best_set):
            best_set = improved_set
            best_size = len(best_set)
            improvement_found = True
            # Optional: Re-run greedy to see if we can fill remaining space?
            # The local search already adds elements.
        
        # If time is running low, break
        if time.time() > deadline - 5:
            break

    # Final verification
    # The construction logic ensures validity, but a double-check is cheap for small N
    if len(best_set) > 0:
        sums = _compute_all_sums(best_set)
        # Calculate expected number of sums
        n = len(best_set)
        expected_sums = (n * (n + 1) * (n + 2)) // 6
        if len(sums) != expected_sums:
             # Fallback to a known valid set if verification fails (should not happen)
             best_set = _greedy_build(n_max)
             
    return best_set
=======
import time
import sys

def _compute_sums_for_set(s):
    """Compute all triple sums for a given set."""
    sums = set()
    n = len(s)
    for i in range(n):
        a = s[i]
        for j in range(i, n):
            b = s[j]
            for k in range(j, n):
                c = s[k]
                sums.add(a + b + c)
    return sums

def _get_new_sums(existing_set, x):
    """Get new sums generated by adding x to existing_set."""
    new_sums = set()
    n = len(existing_set)
    new_sums.add(3 * x)
    for y in existing_set:
        new_sums.add(2 * x + y)
        new_sums.add(x + 2 * y)
    for i in range(n):
        y = existing_set[i]
        for j in range(i + 1, n):
            z = existing_set[j]
            new_sums.add(x + y + z)
    return new_sums

def _can_add_to_sums(existing_sums, existing_set, x):
    """Check if x can be added to existing_set without conflicts."""
    s = 3 * x
    if s in existing_sums:
        return False
    
    for y in existing_set:
        s = 2 * x + y
        if s in existing_sums:
            return False
        s = x + 2 * y
        if s in existing_sums:
            return False
            
    n = len(existing_set)
    for i in range(n):
        y = existing_set[i]
        for j in range(i + 1, n):
            z = existing_set[j]
            s = x + y + z
            if s in existing_sums:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85 # 5s safety margin
    
    n_max = 4096
    max_sum = 3 * n_max
    
    # Global state for backtracking
    used_sums = bytearray(max_sum + 1)
    current_set = []
    sums_generated_by = []
    
    def try_add(x):
        new_sums = []
        
        s = 3 * x
        if used_sums[s]:
            return False
        new_sums.append(s)
        
        for y in current_set:
            s = 2 * x + y
            if used_sums[s]:
                return False
            new_sums.append(s)
            
            s = x + 2 * y
            if used_sums[s]:
                return False
            new_sums.append(s)
                
        n = len(current_set)
        for i in range(n):
            y = current_set[i]
            for j in range(i + 1, n):
                z = current_set[j]
                s = x + y + z
                if used_sums[s]:
                    return False
                new_sums.append(s)
                
        for s in new_sums:
            used_sums[s] = 1
        current_set.append(x)
        sums_generated_by.append(new_sums)
        return True

    def undo_add():
        if not current_set:
            return
        last_sums = sums_generated_by.pop()
        for s in last_sums:
            used_sums[s] = 0
        current_set.pop()

    def dfs(start_val):
        if time.time() > deadline - 2:
            return
        
        for x in range(start_val, n_max + 1):
            if try_add(x):
                dfs(x + 1)
                undo_add()
            
            if time.time() > deadline - 2:
                return

    # 1. Initial Backtracking Search
    # Start with 1
    dfs(1)
    
    best_set = list(current_set)
    best_size = len(best_set)
    
    # 2. Local Search to improve size
    # Try to increase size by 1: Remove 1, Add 2.
    
    def try_improve(set_to_improve, time_limit):
        start_time = time.time()
        current_best = list(set_to_improve)
        current_best_size = len(current_best)
        
        # We'll do a few rounds of local search
        # In each round, we try to swap one element for two new ones.
        
        for _ in range(100): # Max 100 improvement attempts
            if time.time() > time_limit:
                break
                
            improved = False
            n_curr = len(current_best)
            
            # Try removing each element
            for i in range(n_curr):
                if time.time() > time_limit:
                    break
                
                removed_val = current_best[i]
                temp_set = current_best[:i] + current_best[i+1:]
                temp_sums = _compute_sums_for_set(temp_set)
                
                # Try to add 'a'
                found_a = False
                for a in range(1, n_max + 1):
                    if a == removed_val or a in temp_set:
                        continue
                    
                    if _can_add_to_sums(temp_sums, temp_set, a):
                        found_a = True
                        set_with_a = temp_set + [a]
                        set_with_a.sort()
                        sums_with_a = temp_sums.union(_get_new_sums(temp_set, a))
                        
                        # Try to add 'b'
                        for b in range(a + 1, n_max + 1):
                            if b == removed_val or b in set_with_a:
                                continue
                            
                            if _can_add_to_sums(sums_with_a, set_with_a, b):
                                new_set = set_with_a + [b]
                                new_set.sort()
                                if len(new_set) > current_best_size:
                                    current_best = new_set
                                    current_best_size = len(new_set)
                                    improved = True
                                    break
                        if improved:
                            break
                    if time.time() > time_limit:
                        break
                if improved:
                    break
                    
            if not improved:
                # No improvement found in this round, try a different strategy or stop
                # To avoid getting stuck, we can break if no improvement in a full pass
                break
                
        return current_best

    if time.time() < deadline - 5:
        best_set = try_improve(best_set, deadline - 5)

    return best_set
>>>>>>> REPLACE
