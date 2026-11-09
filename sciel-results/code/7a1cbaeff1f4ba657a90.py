import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Helper to generate all valid rows for a given max value M
    def generate_valid_rows(M, k):
        rows = []
        
        def backtrack(current_row, used_diffs):
            if len(current_row) == k + 1:
                # Compute bitmask for this row
                mask = 0
                for i in range(len(current_row)):
                    for j in range(i+1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        mask |= (1 << d)
                rows.append((tuple(current_row), mask))
                return
            
            last = current_row[-1] if current_row else 0
            # Next number must be > last
            # We can limit the range to speed up
            # But for now, let's iterate up to M
            
            for x in range(last + 1, M + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                backtrack(current_row + [x], used_diffs | new_diffs)
        
        backtrack([0], set())
        return rows

    # Initial greedy solution to get a good upper bound
    def greedy_construct(n_rows, k):
        used_diffs = set()
        greedy_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs |= new_diffs
            greedy_rows.append(row)
        return greedy_rows

    best_rows = greedy_construct(n_rows, k)
    best_max = max(max(r) for r in best_rows)
    
    # Try to improve by decreasing M
    # We will try M from best_max - 1 down to some lower bound
    
    # To speed up, we can pre-compute valid rows for a range of M values
    # But M changes, so we have to regenerate.
    
    # Let's try M from best_max - 1 down to 100 (or some lower bound)
    # We can stop early if we find a solution
    
    # To avoid regenerating rows for every M, we can generate rows for the current M
    # and filter them.
    
    # Actually, generating rows is the bottleneck.
    # Let's try a different approach:
    # 1. Generate all valid rows for M = best_max.
    # 2. Search for a solution.
    # 3. If found, update best_max and best_rows.
    # 4. Try M = best_max - 1, etc.
    
    # But generating rows for M=134 might be slow.
    # Let's optimize the row generation.
    
    # Alternative: Use the backtracking search directly, but with a global M constraint.
    # This is similar to the original approach but with a better search strategy.
    
    # Let's stick to the pre-computation approach but with a time limit.
    
    # We'll try M values in descending order.
    # For each M, we generate valid rows and search for a clique of size n_rows.
    
    # To make it faster, we can use the following:
    # - Sort valid rows by their last element.
    # - Use symmetry breaking: only consider rows in a fixed order.
    
    def find_solution(M, n_rows, k):
        valid_rows = generate_valid_rows(M, k)
        if len(valid_rows) < n_rows:
            return None
        
        # Sort by last element, then lexicographically
        valid_rows.sort(key=lambda x: (x[0][-1], x[0]))
        
        # We need to find n_rows rows with disjoint difference masks
        # We can use backtracking
        
        def backtrack(idx, used_mask, chosen_rows):
            if idx == n_rows:
                return chosen_rows
            
            # Try each valid row
            # To break symmetry, we can start from the last index we tried
            # But since rows are sorted, we can just iterate
            for i in range(len(valid_rows)):
                row, mask = valid_rows[i]
                if mask & used_mask:
                    continue
                
                # Check if we can still fit the remaining rows
                # This is a heuristic, not a strict bound
                
                res = backtrack(idx + 1, used_mask | mask, chosen_rows + [row])
                if res is not None:
                    return res
            return None
        
        return backtrack(0, 0, [])
    
    # Main loop: try decreasing M
    # We start from best_max - 1 and go down
    # We can stop if we find a solution
    
    # To avoid spending too much time on a single M, we can set a time limit per M
    # But for now, let's just try a few M values
    
    # We'll try M from best_max - 1 down to max(100, best_max - 20)
    # This is a heuristic to limit the search space
    
    lower_bound = max(100, best_max - 20)
    
    for M in range(best_max - 1, lower_bound - 1, -1):
        if time.time() > deadline:
            break
        
        # Check if M is feasible
        # A necessary condition is that the number of valid rows is at least n_rows
        # But we don't know that without generating them.
        
        res = find_solution(M, n_rows, k)
        if res is not None:
            best_max = M
            best_rows = [list(r) for r in res]
            # Continue to try to find an even smaller M
    
    return best_rows
