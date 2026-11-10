import time

def construct():
    start_time = time.time()
    deadline = 85.0
    n = 12
    
    # Greedy construction with backtracking
    # Build sequence one element at a time, choosing smallest valid value
    # that maintains distinct subset sums
    
    def greedy_construct(n, time_limit):
        """Build a sequence of n distinct positive integers with distinct subset sums,
        minimizing the largest element using greedy with backtracking."""
        
        best_solution = None
        best_max_val = float('inf')
        
        def backtrack(seq, existing_sums, max_val_so_far):
            nonlocal best_solution, best_max_val
            
            if time.time() > time_limit:
                return False
            
            if len(seq) == n:
                current_max = max(seq)
                if current_max < best_max_val:
                    best_max_val = current_max
                    best_solution = seq[:]
                return True
            
            # Try to find the smallest valid next element
            # Start from 1, but we need it to be different from existing elements
            used_values = set(seq)
            
            for candidate in range(1, max_val_so_far + 2):
                if time.time() > time_limit:
                    return False
                
                if candidate in used_values:
                    continue
                
                # Check if adding candidate creates duplicate subset sums
                new_sums = list(existing_sums)
                for s in existing_sums:
                    new_sums.append(s + candidate)
                
                if len(new_sums) != len(set(new_sums)):
                    continue
                
                # Valid candidate, recurse
                seq.append(candidate)
                found = backtrack(seq, new_sums, max(max_val_so_far, candidate))
                seq.pop()
                
                if found:
                    return True
            
            return False
        
        backtrack([], [0], 0)
        return best_solution
    
    # First try greedy construction
    result = greedy_construct(n, deadline - 5.0)
    
    if result is not None:
        return result
    
    # Fallback: powers of 2 (always valid)
    return [1 << i for i in range(n)]
