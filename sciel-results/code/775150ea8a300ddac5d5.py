import time
from math import lcm
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # LCM of 1..22
    L = reduce(lcm, range(1, 23))
    
    # Reciprocals as integers
    # v[i] corresponds to 1/(i+1)
    v = [L // (i + 1) for i in range(22)]
    
    # We want to find the largest k such that there exists a subset of size k
    # that is dissociated.
    
    # Helper to check if we can form a set of size k
    # We use DFS
    # state: idx, current_sums_set, current_count
    # We try to pick elements from v[idx:]
    
    # To speed up, we can sort v?
    # The indices in the final answer must be the original indices 1..22.
    # So we keep track of which original indices are chosen.
    
    # Let's iterate k from 22 down to 1
    for k in range(22, 0, -1):
        # Try to find a subset of size k
        # We can use a recursive search
        # To avoid deep recursion overhead, we can use a stack or just recursion (depth 22 is fine)
        
        found = False
        result_indices = []
        
        def dfs(idx, current_sums, current_count, chosen_indices):
            nonlocal found, result_indices
            if found:
                return
            if time.time() > deadline:
                return
            
            # Pruning: not enough elements left
            if current_count + (22 - idx) < k:
                return
            
            if current_count == k:
                # Check if valid (it should be, due to pruning)
                # But we need to verify all sums are distinct?
                # The pruning ensures that at each step, the size doubled.
                # So if we reached count k, the size is 2^k, which means all sums are distinct.
                found = True
                result_indices = list(chosen_indices)
                return
            
            # Option 1: Skip v[idx]
            dfs(idx + 1, current_sums, current_count, chosen_indices)
            if found:
                return
            
            # Option 2: Include v[idx]
            val = v[idx]
            # Calculate new sums
            # new_sums = {s + val for s in current_sums}
            # Check intersection
            # If current_sums and new_sums are disjoint, we can proceed
            
            # Optimization: if current_sums is empty, new_sums is {val}. Disjoint from empty? Yes.
            # Actually, start with {0}.
            
            # To check intersection efficiently:
            # We can iterate over the smaller set? They are same size.
            # Or just create the set and check `if not current_sums.isdisjoint(new_sums)`
            
            # Creating the set is O(N).
            new_sums = set()
            for s in current_sums:
                new_sums.add(s + val)
            
            if current_sums.isdisjoint(new_sums):
                next_sums = current_sums | new_sums
                chosen_indices.append(idx)
                dfs(idx + 1, next_sums, current_count + 1, chosen_indices)
                chosen_indices.pop()
                if found:
                    return
        
        # Initial call
        # We start with sum {0}
        dfs(0, {0}, 0, [])
        
        if found:
            # result_indices contains indices 0..21
            # Convert to 1..22
            return [i + 1 for i in result_indices]
        
    # Fallback
    return [1]
