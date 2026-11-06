import sys
import time

def construct():
    # Step 1: Greedy baseline to get an initial lower bound
    values_greedy = []
    sums_greedy = set()
    for x in range(1, 257):
        new_sums = []
        current_sum = 0
        for a in reversed([x] + values_greedy):
            current_sum += a
            new_sums.append(current_sum)
        
        if not sums_greedy.intersection(new_sums):
            values_greedy.append(x)
            sums_greedy.update(new_sums)
    
    best_len = len(values_greedy)
    best_seq = values_greedy[:]
    
    # Step 2: CSP with Forward Checking
    # State: 
    #   seq: list of chosen integers
    #   prefix_sums: list of prefix sums where prefix_sums[i] = sum(seq[:i])
    #   forbidden: set of differences between any two prefix sums (i.e., all subarray sums)
    #   max_prefix: the last prefix sum
    
    # Forward Checking:
    # When we add a new element x to the sequence:
    #   new_prefix = max_prefix + x
    #   new subarray sums are: new_prefix - prefix_sums[i] for all i in range(len(prefix_sums)+1)
    #   These must not be in forbidden.
    #   Also, we must ensure that no future element y > x can create a collision with existing forbidden sums
    #   or with the new sums. This is harder to check exactly, but we can prune candidates
    #   if the gap between x and the next available number is too small to avoid collisions.
    
    # A simpler but effective forward checking:
    # For a candidate x, compute all new sums. If any sum is in forbidden, reject x.
    # Additionally, we can look ahead: if we pick x, what constraints does it place on the next element?
    # The next element y > x must satisfy that y + (any suffix sum) is not in forbidden.
    # This is complex. Let's stick to simple forward checking (checking current collision)
    # but use a strong pruning heuristic based on the density of forbidden sums.
    
    # Pruning Heuristic:
    # If the number of forbidden sums is too high relative to the remaining range, we might be stuck.
    # Specifically, if we are at depth d, and the last element is L, the maximum possible additional elements
    # is 256 - L. If d + (256 - L) <= best_len, prune.
    # Also, we can estimate the minimum number of forbidden sums that will be added.
    
    # Let's implement a DFS with these prunings.
    
    # To speed up, we will use a recursive function with careful state management.
    # We will use a time limit to avoid running out of time.
    
    start_time = time.time()
    timeout = 55  # Leave 5 seconds for cleanup/return
    
    # We will use a global variable to track the best solution and allow early termination if we find a very good one
    # or if we exceed the time limit.
    
    # To make the search more effective, we can try to order candidates.
    # Smaller candidates allow more room for future elements, so we try them first.
    
    def dfs(seq, prefix_sums, forbidden, last_val):
        nonlocal best_len, best_seq, start_time, timeout
        
        # Check time
        if time.time() - start_time > timeout:
            return
        
        # Update best
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
        
        # Pruning: Max possible additional elements
        if len(seq) + (256 - last_val) <= best_len:
            return
        
        # If we've already found a sequence of length 256, we can't do better
        if best_len == 256:
            return
        
        # Forward Checking: Try each candidate x > last_val
        # We can optimize by breaking early if we find a collision, but we need to try all valid x
        # to explore the search space.
        
        # Optimization: We can skip candidates that are "too close" to last_val if the forbidden set is dense.
        # But for now, let's just iterate.
        
        # To speed up the collision check, we can pre-compute the suffix sums of the current sequence.
        # suffix_sums[j] = sum of the last j elements.
        # Then the new sums for candidate x are: x + suffix_sums[j] for j in 0..len(seq)
        # (where suffix_sums[0] = 0, suffix_sums[1] = seq[-1], etc.)
        
        # Let's compute suffix sums once per DFS call to avoid recomputation for each x.
        suffix_sums = [0] * (len(seq) + 1)
        for j in range(1, len(seq) + 1):
            suffix_sums[j] = suffix_sums[j-1] + seq[-j]
        
        # Now for each candidate x, the new sums are {x + s for s in suffix_sums}
        
        # We can also pre-compute the set of forbidden sums for O(1) lookup.
        
        for x in range(last_val + 1, 257):
            # Check time
            if time.time() - start_time > timeout:
                return
            
            # Compute new sums
            # new_sums = {x + s for s in suffix_sums}
            # Check if any new sum is in forbidden
            collision = False
            for s in suffix_sums:
                if (x + s) in forbidden:
                    collision = True
                    break
            
            if collision:
                continue
            
            # No collision, so we can proceed
            # New prefix sum
            new_prefix = prefix_sums[-1] + x
            # New forbidden set
            # We need to add all new sums to forbidden
            # To avoid creating a new set object every time, we could use a stack of changes,
            # but for simplicity and given the time limit, let's just create a new set.
            # This might be slow, but let's see.
            
            # Optimization: Instead of creating a new set, we can use a "mark" array or a counter,
            # but sets are hash-based and fast for membership.
            
            # Let's try to optimize by only adding the new sums to a temporary set and checking
            # if the union would be valid. But we need to pass the updated forbidden set to the next call.
            
            # Alternative: Use a global array `forbidden_count` of size (max_possible_sum + 1)
            # and increment/decrement as we recurse. This avoids set creation.
            # Max possible sum is 256 * 256 = 65536.
            # Let's use this approach.
            
            # But we already have `forbidden` as a set. Let's stick with sets for now
            # and see if it's fast enough. If not, we'll switch to the array approach.
            
            # For now, let's just create the new set.
            new_forbidden = forbidden | set(x + s for s in suffix_sums)
            new_prefix_sums = prefix_sums + [new_prefix]
            
            dfs(seq + [x], new_prefix_sums, new_forbidden, x)
            
            # If we found a solution of length 256, we can stop
            if best_len == 256:
                return
        
        return
    
    # Start DFS
    # Initial state: empty sequence, prefix_sums = [0], forbidden = empty set, last_val = 0
    dfs([], [0], set(), 0)
    
    return best_seq
