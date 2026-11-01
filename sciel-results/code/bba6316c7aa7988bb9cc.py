import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows and k differences (k+1 elements per row).
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    deadline = 115.0  # 15s margin before 130s limit (or 120s if strictly enforced, let's use 110s to be safe)
    
    # Fallback: Greedy solution (always valid)
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_solution = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_solution)
    
    # Target scope: We know 112 is possible for (7,5).
    # Let's try to find a solution with scope 112, then 113, etc.
    # If we find one, we update best_solution.
    
    # We will use a backtracking search.
    # State: list of rows, set of used differences
    # We try to fill n_rows rows.
    
    def is_valid_solution(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def backtrack(rows, used_diffs, target_scope):
        nonlocal best_solution, best_scope, deadline, start_time
        
        if len(rows) == n_rows:
            # Valid solution found
            current_scope = max(max(row) for row in rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_solution = [list(r) for r in rows]
            return True
        
        if time.time() > deadline:
            return False
            
        # Symmetry breaking: The rows are indistinguishable. 
        # We can enforce that the last element of the current row is <= the last element of the next row?
        # Actually, a stronger constraint: We can sort the rows by their last element at the end.
        # During search, we can enforce that the last element of the new row is >= the last element of the previous row?
        # No, that doesn't break all symmetry, but it helps.
        # Let's enforce: last element of row i <= last element of row i+1.
        
        prev_last = rows[-1][-1] if rows else -1
        
        # We need to choose k new elements (since 0 is fixed) for the new row.
        # The row must be increasing: 0 < a1 < a2 < ... < ak <= target_scope
        # And all differences must be new.
        
        # Heuristic: Try to find a row that uses small differences?
        # Let's just iterate through possible rows.
        
        # To speed up, we can generate candidate rows.
        # A row is defined by its k largest elements.
        
        # We'll use a recursive helper to build the single row.
        def build_row(current_row, used_diffs_local, target_scope, min_val):
            nonlocal deadline, start_time
            
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                # Check if this row can be added
                # Differences are already checked in build_row
                rows.append(current_row)
                if backtrack(rows, used_diffs_local, target_scope):
                    return True
                rows.pop()
                return False
            
            # Determine next element
            # It must be > current_row[-1] and <= target_scope
            # Also, its difference with all previous elements in current_row must not be in used_diffs_local
            
            # Optimization: The smallest possible next value
            last_val = current_row[-1]
            # We can start from last_val + 1
            # But we can prune: if the remaining slots cannot fit in the remaining space?
            # Remaining slots = k + 1 - len(current_row)
            # Max value = target_scope
            # Min increment = 1
            # So we need (target_scope - last_val) >= remaining_slots - 1
            
            remaining_slots = k + 1 - len(current_row)
            if target_scope - last_val < remaining_slots - 1:
                return False
                
            start_val = last_val + 1
            # Upper bound for this element:
            # We need to leave room for the rest.
            # Max val for current pos = target_scope - (remaining_slots - 1)
            end_val = target_scope - (remaining_slots - 1)
            
            for v in range(start_val, end_val + 1):
                # Check differences
                new_diffs = {v - a for a in current_row}
                if new_diffs & used_diffs_local:
                    continue
                
                # Recurse
                current_row.append(v)
                new_used = used_diffs_local | new_diffs
                if build_row(current_row, new_used, target_scope, v):
                    return True
                current_row.pop()
                
                if time.time() > deadline:
                    return False

        # Initial call for the first element of the new row
        # The row starts with 0.
        # We need to pick k more elements.
        # We can start the search for the new row.
        
        # To reduce symmetry between rows, we can enforce that the last element of the new row is >= prev_last.
        # This is handled by the fact that we are building rows sequentially and we will sort them?
        # Actually, if we just search, we might find permutations.
        # Enforcing last element >= prev_last reduces permutations by n! roughly.
        
        # Let's modify build_row to accept a constraint on the last element?
        # Or just check at the end.
        # It's better to prune early.
        
        # Let's just call build_row. The symmetry breaking will be weak, but for small N it's ok.
        # Wait, if we don't break symmetry, we might waste time.
        # Let's enforce: The last element of the new row must be >= prev_last.
        
        # We can integrate this into the build_row logic by passing a min_last_val constraint.
        # But build_row builds the whole row.
        
        # Simpler: Just run build_row. If it finds a row with last element < prev_last, we skip it?
        # No, that's inefficient.
        
        # Let's assume the rows are ordered by their max value.
        # We will pass a `min_last_val` to the row builder.
        
        def build_row_constrained(current_row, used_diffs_local, target_scope, min_last_val):
            nonlocal deadline, start_time
            
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                if current_row[-1] < min_last_val:
                    return False
                rows.append(current_row)
                if backtrack(rows, used_diffs_local, target_scope):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            remaining_slots = k + 1 - len(current_row)
            if target_scope - last_val < remaining_slots - 1:
                return False
                
            start_val = last_val + 1
            end_val = target_scope - (remaining_slots - 1)
            
            for v in range(start_val, end_val + 1):
                new_diffs = {v - a for a in current_row}
                if new_diffs & used_diffs_local:
                    continue
                
                current_row.append(v)
                new_used = used_diffs_local | new_diffs
                if build_row_constrained(current_row, new_used, target_scope, min_last_val):
                    return True
                current_row.pop()
                
                if time.time() > deadline:
                    return False
            return False

        # Call for the new row
        # The new row must have last element >= prev_last
        build_row_constrained([0], used_diffs, target_scope, prev_last)
        
        return False

    # Try scopes starting from the theoretical lower bound or known optimal
    # Lower bound: Total diffs = n_rows * k * (k+1) / 2.
    # For n=7, k=5: 7 * 15 = 105.
    # So scope >= 105.
    
    # We know 112 is possible.
    # Let's try 112 first.
    
    for target_scope in range(105, 120):
        if time.time() > deadline:
            break
            
        rows = []
        used = set()
        # Start backtracking
        # Note: The symmetry breaking (last element >= prev_last) assumes rows are sorted by max.
        # This is a valid restriction because we can always sort the final set of rows by their max element.
        
        # We need to reset the state for each target_scope
        # The backtrack function modifies `rows` and `used`. We pass `rows` and `used` as arguments or use nonlocal.
        # In the code above, `backtrack` uses `rows` and `used_diffs` from the outer scope or arguments?
        # I defined `backtrack(rows, used_diffs, target_scope)`.
        # Inside `backtrack`, I call `build_row_constrained` which appends to `rows`.
        # This looks correct.
        
        # However, `backtrack` is recursive and modifies `rows`.
        # We must ensure `rows` is empty at the start of each `target_scope` attempt.
        
        if backtrack([], set(), target_scope):
            # Found a solution
            break
            
    return best_solution
# EVOLVE-BLOCK-END
