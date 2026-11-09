import time
import random
import sys

def construct():
    """
    Finds a maximum-size Sidon set in [1, 284] containing {1, 2, 5, 11}.
    Uses a CSP approach with randomized backtracking and forward checking.
    """
    deadline = time.time() + 145.0  # Safety margin from 150s
    fixed = [1, 2, 5, 11]
    N = 284
    fixed_set = set(fixed)
    
    # We aim for size 20. If we can't find a valid one of size 20, we try 19, 18, etc.
    # The optimal size without constraints is 20. With constraints, it might be 19 or 20.
    # We will try to find size 20 first.
    
    def solve_csp(target_size, time_limit):
        """
        Solves for a Sidon set of exactly target_size.
        Returns the set if found, else None.
        """
        start_time = time.time()
        
        # Variables: indices 0 to target_size - 1
        # We assign values to these variables.
        # To help constraint propagation, we can treat the fixed elements as pre-assigned.
        
        # Let's structure the assignment:
        # First, assign the fixed elements.
        # Then assign the remaining (target_size - len(fixed)) elements.
        
        # To make backtracking efficient, we should assign variables in an order that
        # maximizes constraint propagation. However, standard backtracking assigns x0, x1, ...
        # We can use a heuristic: assign variables that have the most constrained domains first?
        # Or just randomize the order of non-fixed variables to avoid bias?
        
        # Let's use a recursive backtracking with a stack to allow random restarts within the call stack?
        # No, simple recursion with a time check is fine, but we need to handle the "random restart" logic.
        # Better: Implement a loop that performs a single backtracking search, and if it fails,
        # restart with a different random seed/order.
        
        # Actually, a single deep backtracking search might get stuck.
        # Let's do randomized backtracking with a step limit per search, and restart if stuck.
        
        # We'll define a helper that performs a DFS with forward checking.
        
        def dfs(assignments, used_sums, remaining_vars):
            """
            assignments: dict var_idx -> value
            used_sums: set of all sums a+b for a,b in assignments (a<=b)
            remaining_vars: list of var indices not yet assigned
            """
            if time.time() > deadline - 2.0:
                return None, True # Timeout flag
            
            if not remaining_vars:
                return assignments, False
            
            # Select variable with Minimum Remaining Values (MRV) heuristic?
            # Calculating MRV requires checking all domains against used_sums.
            # This is expensive but prunes heavily.
            
            # For simplicity and speed, let's just pick a random variable from remaining_vars?
            # Or pick the one with the smallest domain?
            
            # Let's try picking a variable randomly to introduce diversity, 
            # but order the domain values randomly.
            
            var = random.choice(remaining_vars)
            remaining_vars.remove(var)
            
            # Generate domain for this variable
            # Domain is [1, N] excluding values already in assignments
            assigned_vals = set(assignments.values())
            
            # We can filter the domain based on used_sums to do forward checking?
            # Forward checking: for a candidate value v, check if v+v or v+u (for all u in assignments)
            # creates a conflict with used_sums. If so, v is not allowed.
            
            # To speed up, we can pre-filter the domain.
            # But the domain size is up to 284. Filtering 284 * len(assignments) is fast.
            
            valid_candidates = []
            for v in range(1, N + 1):
                if v in assigned_vals:
                    continue
                
                # Check conflicts with existing assignments
                # 1. v + v
                if v + v in used_sums:
                    continue
                
                # 2. v + u for all u in assignments
                conflict = False
                for u in assignments.values():
                    if v + u in used_sums:
                        conflict = True
                        break
                if conflict:
                    continue
                
                valid_candidates.append(v)
            
            if not valid_candidates:
                return None, False # Dead end
            
            # Shuffle candidates to randomize the search path
            random.shuffle(valid_candidates)
            
            for v in valid_candidates:
                # Try assigning v to var
                new_assignments = assignments.copy()
                new_assignments[var] = v
                
                # Update used_sums
                new_used_sums = used_sums.copy()
                new_used_sums.add(v + v)
                for u in assignments.values():
                    new_used_sums.add(v + u)
                
                result, timeout = dfs(new_assignments, new_used_sums, remaining_vars[:])
                if timeout:
                    return None, True
                if result is not None:
                    return result, False
            
            return None, False # No candidate worked
        
        # Initial setup for CSP
        # We have target_size variables.
        # Let's map the fixed elements to specific variable indices to simplify?
        # Or just let the solver assign any variable to any value, but enforce that
        # the final set contains the fixed elements?
        
        # Better approach: Pre-assign the fixed elements to the first len(fixed) variables.
        # This reduces the search space.
        
        init_assignments = {}
        init_used_sums = set()
        
        # Assign fixed elements to vars 0, 1, 2, 3
        # Note: The order of fixed elements in the set doesn't matter for sums,
        # but we need to ensure they are distinct. They are.
        for i, f in enumerate(fixed):
            init_assignments[i] = f
        
        # Calculate initial sums
        for i in range(len(fixed)):
            for j in range(i, len(fixed)):
                init_used_sums.add(fixed[i] + fixed[j])
        
        remaining_vars = list(range(len(fixed), target_size))
        
        # Run DFS
        result, timeout = dfs(init_assignments, init_used_sums, remaining_vars)
        
        if result is not None:
            # Extract the set of values
            return sorted(result.values())
        
        return None

    best_found = None
    best_size = 0
    
    # Try to find a set of size 20 first
    # We will run multiple randomized backtracking searches.
    # Each search is a call to solve_csp, but solve_csp as defined above is a single DFS path?
    # No, DFS explores the tree. But it's deterministic if we don't shuffle.
    # We DO shuffle candidates in dfs, so each call to solve_csp is a randomized search.
    # However, DFS might go very deep and get stuck in a bad branch.
    # It's better to have a step limit per DFS and restart.
    
    # Let's modify the strategy:
    # Run a loop. In each iteration, perform a randomized backtracking search with a step limit.
    # If it succeeds, return.
    # If it fails or times out, restart.
    
    # To implement step limit in DFS, we need to pass a counter or check time.
    # Checking time is easier and sufficient.
    
    # Let's try size 20.
    for size in [20, 19, 18, 17, 16]:
        if time.time() > deadline - 5.0:
            break
            
        found = None
        # Try multiple random restarts for this size
        # Each restart is a fresh DFS.
        # We'll allow each DFS to run for a small chunk of time, say 2-5 seconds?
        # Or just let it run until it finds a solution or times out globally?
        # The DFS itself has the global deadline check.
        # If we just call solve_csp(size, ...), it will run until it finds a solution or hits the global deadline.
        # This is risky if the first DFS gets stuck in a huge subtree.
        
        # Better: Implement a "node limit" or "time slice" for each DFS attempt.
        
        def solve_csp_with_time_slice(target_size, time_slice):
            start = time.time()
            init_assignments = {}
            init_used_sums = set()
            for i, f in enumerate(fixed):
                init_assignments[i] = f
            for i in range(len(fixed)):
                for j in range(i, len(fixed)):
                    init_used_sums.add(fixed[i] + fixed[j])
            remaining_vars = list(range(len(fixed), target_size))
            
            # Recursive DFS with time slice check
            def dfs(assignments, used_sums, remaining_vars):
                if time.time() - start > time_slice:
                    return None, 'timeout'
                if time.time() > deadline - 2.0:
                    return None, 'global_timeout'
                
                if not remaining_vars:
                    return assignments, 'success'
                
                # MRV Heuristic: Find variable with fewest valid candidates
                # This is O(V * D * A) which might be slow if V is large.
                # V is at most 16. D is 284. A is at most 16.
                # 16 * 284 * 16 ~ 72k ops per node. This is fast enough.
                
                best_var = None
                best_candidates = None
                min_count = float('inf')
                
                for var in remaining_vars:
                    assigned_vals = set(assignments.values())
                    valid_candidates = []
                    for v in range(1, N + 1):
                        if v in assigned_vals:
                            continue
                        if v + v in used_sums:
                            continue
                        conflict = False
                        for u in assignments.values():
                            if v + u in used_sums:
                                conflict = True
                                break
                        if conflict:
                            continue
                        valid_candidates.append(v)
                    
                    count = len(valid_candidates)
                    if count < min_count:
                        min_count = count
                        best_var = var
                        best_candidates = valid_candidates
                        if count == 0:
                            return None, 'dead_end'
                        if count == 1:
                            break # Can't get better than 1
                
                if best_var is None:
                    return None, 'dead_end'
                
                remaining_vars.remove(best_var)
                
                # Shuffle candidates
                random.shuffle(best_candidates)
                
                for v in best_candidates:
                    new_assignments = assignments.copy()
                    new_assignments[best_var] = v
                    new_used_sums = used_sums.copy()
                    new_used_sums.add(v + v)
                    for u in assignments.values():
                        new_used_sums.add(v + u)
                    
                    result, status = dfs(new_assignments, new_used_sums, remaining_vars[:])
                    if status == 'success':
                        return result, 'success'
                    elif status == 'global_timeout':
                        return None, 'global_timeout'
                    elif status == 'timeout':
                        return None, 'timeout'
                    # else dead_end, continue to next candidate
                
                return None, 'dead_end'
            
            result, status = dfs(init_assignments, init_used_sums, remaining_vars)
            if status == 'success':
                return sorted(result.values())
            return None

        # Run multiple attempts for this size
        attempts = 0
        while time.time() < deadline - 5.0:
            attempts += 1
            # Give each attempt a time slice, e.g., 1 second?
            # If MRV is used, it should be fast.
            res = solve_csp_with_time_slice(size, 2.0)
            if res:
                best_found = res
                best_size = size
                break
        
        if best_found:
            break
        
    if best_found:
        return best_found
    
    # Fallback: Return a valid set of smaller size if found, or a greedy set
    # If CSP failed completely, try a simple greedy construction to ensure we return *something* valid
    # Although CSP should find size 16+ easily.
    
    # Greedy Fallback
    current_set = list(fixed)
    current_set.sort()
    used_sums = set()
    for i in range(len(current_set)):
        for j in range(i, len(current_set)):
            used_sums.add(current_set[i] + current_set[j])
            
    for x in range(N, 0, -1):
        if x in current_set:
            continue
        valid = True
        if x + x in used_sums:
            valid = False
        else:
            for y in current_set:
                if x + y in used_sums:
                    valid = False
                    break
        if valid:
            current_set.append(x)
            current_set.sort()
            used_sums.add(x + x)
            for y in current_set[:-1]:
                used_sums.add(x + y)
                
    return sorted(current_set)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(f"Set: {res}")
    # Verify
    sums = set()
    valid = True
    for i in range(len(res)):
        for j in range(i, len(res)):
            s = res[i] + res[j]
            if s in sums:
                valid = False
            sums.add(s)
    print(f"Valid: {valid}")
    # Check fixed
    print(f"Contains fixed: {set([1, 2, 5, 11]).issubset(set(res))}")

