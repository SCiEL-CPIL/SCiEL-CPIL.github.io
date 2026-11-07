import time
import sys

sys.setrecursionlimit(10000)

DEADLINE = 85.0  # safety margin from 90s kill

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # We want to find the maximum size k such that there exists a valid set of size k.
    # We know size 7 is possible. Let's aim for 8, then 9.
    
    # To speed up, we can use a precomputed list of candidates.
    # A good heuristic is to try larger numbers first? Or specific patterns?
    # Actually, random restarts with a good local search or backtracking is best.
    
    # Let's use a backtracking approach that tries to build a set of size K.
    # We will try K from 9 down to 7. If we find one, we return it.
    # If we don't find one in time, we return the best found (likely 7 or 8).
    
    best_set = [31, 32, 34] # Fallback size 3? No, we know 7 is possible.
    # Let's use a known size 7 set as fallback.
    # {1, 2, 4, 8, 16, 32, 64} is invalid because 1 divides all.
    # {2, 4, 8, 16, 32, 64, 128} -> sums are 2..254 even. 2 divides 4, etc. Invalid.
    # We need a set where no sum divides another.
    # Example: {5, 11, 17, 23, 29, 35, 41}? 
    # Let's just run the search.
    
    # Optimization: Use a boolean array for sums to check divisibility quickly.
    # Max sum for 8 elements of 256 is 2048.
    # Max sum for 9 elements is 2304.
    # Let's allocate for 4096 to be safe.
    
    MAX_SUM = 4096
    is_sum = [False] * (MAX_SUM + 1)
    
    def check_validity(S, is_sum_local):
        """
        Check if S is valid.
        is_sum_local is the bitset of sums of S.
        Returns True if valid.
        """
        # Check divisibility
        # For each sum s, check if 2s, 3s... are in is_sum_local
        for s in range(1, MAX_SUM + 1):
            if is_sum_local[s]:
                for k in range(2, MAX_SUM // s + 1):
                    if is_sum_local[s * k]:
                        return False
        return True

    def try_add(S, x, is_sum_local, sums_list):
        """
        Try to add x to S.
        Returns (new_is_sum, new_sums_list) if valid, else (None, None).
        """
        # New sums are sums_list + {s + x for s in sums_list}
        # Check for collision (uniqueness)
        new_sums = []
        new_is_sum = is_sum_local[:] # Copy is expensive? 
        # Actually, let's just compute the new sums and check.
        
        # To avoid copying the whole array, we can check conflicts on the fly.
        # But we need the new state for recursion.
        # Copying a list of 4096 bools is fast.
        
        new_sums_set = set(sums_list)
        for s in sums_list:
            ns = s + x
            if ns in new_sums_set:
                return None, None # Collision
            new_sums_set.add(ns)
        
        new_sums_list = list(new_sums_set)
        
        # Build new is_sum
        new_is_sum = is_sum_local[:]
        for s in new_sums_list:
            if s in sums_list:
                continue # Already there
            new_is_sum[s] = True
            # Check if this new sum creates a divisibility conflict with existing or other new sums
            # Actually, we should check all pairs in the new set.
            
        # Check divisibility on the new set
        # This is O(N^2) roughly if not careful.
        # Better: iterate through all sums in new_sums_list, check multiples.
        for s in new_sums_list:
            for k in range(2, MAX_SUM // s + 1):
                if new_is_sum[s * k]:
                    return None, None
        # Also check if any existing sum divides a new sum?
        # The loop above checks if s * k is in the set.
        # If s is old and s*k is new, it's caught.
        # If s is new and s*k is old, it's caught.
        # If both are new, it's caught.
        
        return new_is_sum, new_sums_list

    def backtrack(S, is_sum_local, sums_list, target_size, deadline):
        if len(S) == target_size:
            return S
        
        if time.time() > deadline:
            return None
        
        # Heuristic: try candidates in a specific order
        # To maximize chance of finding a valid set, we can try candidates that are "far" from existing sums
        # or just iterate 1..256.
        
        # Optimization: If we have used up all candidates, return None
        # We can pass a 'start' index to avoid re-trying small numbers if we sort?
        # But the set elements don't need to be sorted in the final output, 
        # but for backtracking, we can assume we pick them in increasing order to avoid permutations.
        
        # Let's iterate x from 1 to 256
        for x in range(1, 257):
            if x in S:
                continue
            
            new_is_sum, new_sums_list = try_add(S, x, is_sum_local, sums_list)
            if new_is_sum is not None:
                result = backtrack(S + [x], new_is_sum, new_sums_list, target_size, deadline)
                if result is not None:
                    return result
        return None

    # Try to find size 9
    # If fails, try 8
    # If fails, try 7
    
    # Initial state
    initial_is_sum = [False] * (MAX_SUM + 1)
    initial_sums = [0] # Include 0 for ease of calculation? 
    # The problem says NONEMPTY subset sums.
    # But for the uniqueness check, including 0 helps: 
    # If s1+x == s2+x, then s1==s2.
    # If we include 0, sums are {0} U {non-empty sums}.
    # Divisibility check: 0 is not a divisor. 0 is not in non-empty sums.
    # So we can include 0 in our internal representation to make addition easier.
    # But we must ensure 0 is not considered a "sum" for divisibility (0 divides nothing, nothing divides 0 in this context usually, but 0 is not a positive integer sum).
    # The sums are positive integers.
    
    initial_is_sum[0] = True
    
    # We want to maximize size.
    # Let's try 9 first.
    # To make it faster, we can use a randomized candidate order or specific heuristics.
    # But simple backtracking with pruning might be slow for 9.
    
    # Let's try a few restarts with randomized candidate order for size 9.
    import random
    
    found_9 = None
    found_8 = None
    
    # Try size 9 with a few seconds budget
    t_start = time.time()
    for _ in range(50): # 50 restarts
        if time.time() > deadline - 10:
            break
        
        # Randomize candidate order? 
        # Or just use the standard order but with a different random seed for tie-breaking?
        # Actually, the order of x in the loop matters.
        # Let's shuffle the range 1..256 for each restart?
        # That changes the search path.
        
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        # We need to adapt backtrack to use a specific order.
        # Let's modify backtrack to take a candidate list.
        
        def backtrack_ordered(S, is_sum_local, sums_list, target_size, candidates, deadline):
            if len(S) == target_size:
                return S
            
            if time.time() > deadline:
                return None
            
            for x in candidates:
                if x in S:
                    continue
                
                new_is_sum, new_sums_list = try_add(S, x, is_sum_local, sums_list)
                if new_is_sum is not None:
                    result = backtrack_ordered(S + [x], new_is_sum, new_sums_list, target_size, candidates, deadline)
                    if result is not None:
                        return result
            return None

        # Use a subset of candidates to speed up? 
        # No, we need all 1..256.
        
        # To avoid deep recursion overhead, we can limit depth or use iterative deepening?
        # Backtracking depth is 9, which is small.
        
        # However, checking `x in S` is O(N). We can use a set for S.
        # But S is small.
        
        # Let's run one search for size 9.
        # If it's too slow, we break.
        
        # Actually, shuffling 256 items and backtracking might be slow if it explores many dead ends.
        # A better heuristic: Sort candidates by some score?
        # For now, let's just run it.
        
        res = backtrack_ordered([], initial_is_sum, [0], 9, candidates, deadline - 5)
        if res:
            found_9 = res
            break
    
    if found_9:
        return found_9
    
    # Try size 8
    for _ in range(100):
        if time.time() > deadline - 2:
            break
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        res = backtrack_ordered([], initial_is_sum, [0], 8, candidates, deadline - 1)
        if res:
            found_8 = res
            break
    
    if found_8:
        return found_8
    
    # Try size 7
    for _ in range(200):
        if time.time() > deadline:
            break
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        res = backtrack_ordered([], initial_is_sum, [0], 7, candidates, deadline)
        if res:
            return res
            
    # Fallback
    return [31, 32, 34]

# Wait, the fallback [31, 32, 34] might be invalid?
# Sums: 31, 32, 34, 63, 65, 66, 97.
# 31 divides 63? No. 31*2=62.
# 32 divides 64? 64 is not a sum.
# 34 divides 68? No.
# 63 divides ...? No.
# It is likely valid.
# But we should ensure the fallback is valid.

# Let's refine the `try_add` to be faster.
# Copying `is_sum` (size 4096) at every step is O(4096).
# Depth 9 -> 9 * 4096 operations per path.
# Number of paths? Could be large.

# Alternative: Use a single global `is_sum` and undo changes (backtracking).
# This is much faster.

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MAX_SUM = 4096
    is_sum = [False] * (MAX_SUM + 1)
    is_sum[0] = True
    current_sums = [0]
    
    def try_add_fast(x):
        """
        Adds x to the current state.
        Returns True if valid, False otherwise.
        Modifies is_sum and current_sums in place.
        Caller must call undo_add(x) if it wants to revert.
        """
        new_sums = []
        # Check for collision
        for s in current_sums:
            ns = s + x
            if is_sum[ns]:
                return False
            new_sums.append(ns)
        
        # Check for divisibility conflicts
        # New sums are new_sums.
        # Check if any new sum divides or is divided by any existing sum or other new sum.
        
        # 1. Check new sums against each other and existing sums
        # For each ns in new_sums:
        #   Check if ns * k is in is_sum (existing or previously added new sums)
        #   Check if ns / k is in is_sum (integer division)
        
        # To do this efficiently, we can just mark them and then check?
        # Or check on the fly.
        
        # Let's mark them first? No, if we mark them, we can check against them.
        # But if we find a conflict, we need to unmark.
        
        # Better: Check conflicts before marking.
        # For each ns in new_sums:
        #   For k in 2..MAX_SUM//ns:
        #      if is_sum[ns*k]: return False
        #   For k in 2..MAX_SUM//ns:
        #      if ns % k == 0 and is_sum[ns//k]: return False
        #      # Note: ns//k is a divisor. If ns//k is in is_sum, then (ns//k) divides ns.
        #      # This is covered by the first check if we iterate over divisors?
        #      # If a | b, then b = a * k.
        #      # So checking if ns * k is in is_sum covers "ns divides something".
        #      # Checking if ns / k is in is_sum covers "something divides ns".
        #      # Actually, just checking if any multiple of ns is in is_sum is NOT enough.
        #      # We need: for any a, b in Sums, a !| b and b !| a.
        #      # So for each new sum ns, we check:
        #      # 1. Does ns divide any existing sum? (ns * k in is_sum)
        #      # 2. Does any existing sum divide ns? (ns / k in is_sum, i.e., ns % k == 0 and is_sum[ns//k])
        #      # 3. Does ns divide any other new sum? (We can check pairwise or mark and check)
        #      # 4. Does any other new sum divide ns?
        
        # To handle new vs new, we can just collect new_sums, and for each pair (ns1, ns2), check divisibility.
        # Or, mark new_sums in a temporary set, and check against is_sum U temp_set.
        
        temp_set = set(new_sums)
        
        for ns in new_sums:
            # Check against existing
            for k in range(2, MAX_SUM // ns + 1):
                if is_sum[ns * k]:
                    return False
            # Check if existing divides ns
            # Iterate divisors of ns? Or just check all k?
            # If ns is large, checking all k up to ns/2 is slow.
            # But ns <= 2048.
            for k in range(2, ns + 1):
                if ns % k == 0 and is_sum[ns // k]:
                    return False
            
            # Check against other new sums
            for ns2 in temp_set:
                if ns == ns2:
                    continue
                # ns divides ns2?
                if ns2 % ns == 0:
                    return False
                # ns2 divides ns?
                if ns % ns2 == 0:
                    return False
        
        # If all checks pass, commit
        for ns in new_sums:
            is_sum[ns] = True
            current_sums.append(ns)
        return True

    def undo_add(x):
        """
        Reverts the addition of x.
        """
        # Remove x from current_sums? No, current_sums contains all sums.
        # We need to remove the sums that were added.
        # The sums added are {s + x for s in old_sums}.
        # We can just iterate current_sums and remove those > max_old_sum?
        # Or we can store the old sums list.
        
        # Simpler: We know the new sums are the ones added.
        # But we didn't store which ones specifically.
        # We can recompute?
        # Or, we can just set is_sum[ns] = False for all ns that are > max(old_sums)?
        # No, that's not unique.
        
        # Let's change strategy: Store the list of sums added in a global variable or pass it.
        # Or, simply, in try_add_fast, we can store the new sums in a global list `added_sums`.
        
        # Let's refactor to use a global `added_sums` list.
        
        pass # Placeholder

    # Refactored approach:
    # Use a stack of sums.
    
    global added_sums
    added_sums = []

    def try_add_fast_v2(x):
        new_sums = []
        for s in current_sums:
            ns = s + x
            if is_sum[ns]:
                return False
            new_sums.append(ns)
        
        temp_set = set(new_sums)
        
        for ns in new_sums:
            for k in range(2, MAX_SUM // ns + 1):
                if is_sum[ns * k]:
                    return False
            for k in range(2, ns + 1):
                if ns % k == 0 and is_sum[ns // k]:
                    return False
            for ns2 in temp_set:
                if ns == ns2: continue
                if ns2 % ns == 0: return False
                if ns % ns2 == 0: return False
        
        for ns in new_sums:
            is_sum[ns] = True
            current_sums.append(ns)
            added_sums.append(ns)
        return True

    def undo_add_v2(x):
        # Remove the last len(added_sums) sums?
        # No, added_sums contains all sums added in this step.
        # We need to clear added_sums and remove from current_sums and is_sum.
        
        # But current_sums is a list. Appending is easy, removing specific items is hard.
        # We can just rebuild current_sums?
        # Or, we can maintain current_sums as a set?
        # Iterating a set is slower than list?
        
        # Let's use a list and just pop the last N items?
        # The order of addition is consistent.
        # When we add x, we append new_sums.
        # When we undo, we pop len(new_sums) items.
        # But we need to know len(new_sums).
        # We can store the count in added_sums?
        # Or just store the sums in a local list in try_add, and return it?
        
        # Let's modify try_add to return the list of added sums.
        
        pass

    # Let's restart the implementation with a cleaner state management.
    
    state_sums = [0]
    state_is_sum = [False] * (MAX_SUM + 1)
    state_is_sum[0] = True
    
    def try_add(x):
        nonlocal state_sums, state_is_sum
        new_sums = []
        for s in state_sums:
            ns = s + x
            if state_is_sum[ns]:
                return None
            new_sums.append(ns)
        
        temp_set = set(new_sums)
        
        for ns in new_sums:
            for k in range(2, MAX_SUM // ns + 1):
                if state_is_sum[ns * k]:
                    return None
            for k in range(2, ns + 1):
                if ns % k == 0 and state_is_sum[ns // k]:
                    return None
            for ns2 in temp_set:
                if ns == ns2: continue
                if ns2 % ns == 0: return None
                if ns % ns2 == 0: return None
        
        for ns in new_sums:
            state_is_sum[ns] = True
            state_sums.append(ns)
        return new_sums

    def undo_add(added_sums_list):
        nonlocal state_sums, state_is_sum
        for ns in added_sums_list:
            state_is_sum[ns] = False
            state_sums.remove(ns) # O(N) removal. N is small.

    def backtrack(S, target_size, deadline):
        if len(S) == target_size:
            return S
        
        if time.time() > deadline:
            return None
        
        for x in range(1, 257):
            if x in S:
                continue
            
            added = try_add(x)
            if added is not None:
                result = backtrack(S + [x], target_size, deadline)
                if result is not None:
                    return result
                undo_add(added)
        
        return None

    import random
    
    # Try size 9
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        # Randomize order?
        # The loop `for x in range(1, 257)` is fixed.
        # To get different paths, we can shuffle the range.
        # But `backtrack` uses `range(1, 257)`.
        # Let's pass the candidate list.
        
        def backtrack_with_order(S, target_size, candidates, deadline):
            if len(S) == target_size:
                return S
            if time.time() > deadline:
                return None
            
            for x in candidates:
                if x in S:
                    continue
                added = try_add(x)
                if added is not None:
                    result = backtrack_with_order(S + [x], target_size, candidates, deadline)
                    if result is not None:
                        return result
                    undo_add(added)
            return None

        candidates = list(range(1, 257))
        random.shuffle(candidates)
        res = backtrack_with_order([], 9, candidates, deadline - 5)
        if res:
            return res

    # Try size 8
    for _ in range(50):
        if time.time() > deadline - 2:
            break
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        res = backtrack_with_order([], 8, candidates, deadline - 1)
        if res:
            return res

    # Try size 7
    for _ in range(100):
        if time.time() > deadline:
            break
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        res = backtrack_with_order([], 7, candidates, deadline)
        if res:
            return res

    return [31, 32, 34]

# One issue: `state_sums.remove(ns)` is O(N).
# N can be up to 2^7 = 128 for size 7.
# 128 * 100 restarts * 256 candidates * depth... might be slow.
# But depth is small.

# Let's verify the fallback.
# [31, 32, 34]
# Sums: 31, 32, 34, 63, 65, 66, 97.
# 31 | 63? No.
# 32 | 64? No.
# 34 | 68? No.
# 63 | ...? No.
# It is valid.

# Final check on types.
# Return list of ints.

