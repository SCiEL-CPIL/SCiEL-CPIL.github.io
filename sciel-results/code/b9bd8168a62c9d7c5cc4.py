import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill
    
    num_elements = k + 1
    total_diffs_per_row = num_elements * (num_elements - 1) // 2
    
    # The task specifies n_rows=7, k=6. 
    # Best known scope is 112. Counting bound is 105.
    # We will target scopes from 105 upwards.
    
    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def is_valid(rows):
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_diffs(r))
        return len(all_diffs) == len(set(all_diffs))

    def greedy_row(used_diffs, max_val):
        """
        Greedily construct a valid row starting with 0, using smallest possible values
        that don't conflict with used_diffs.
        """
        row = [0]
        current_max = 0
        while len(row) < num_elements:
            found = False
            for v in range(current_max + 1, max_val + 1):
                # Check if adding v creates any duplicate diff
                valid = True
                for x in row:
                    d = v - x
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    # Check internal consistency (though greedy ensures this if we add one by one? 
                    # No, we must ensure v-x are distinct from each other too, but they are distinct 
                    # because x are distinct and v is fixed. 
                    # However, we must ensure v-x is not in used_diffs.
                    row.append(v)
                    for x in row[:-1]:
                        used_diffs.add(v - x)
                    current_max = v
                    found = True
                    break
            if not found:
                return None
        return row

    def solve_with_greedy(max_scope):
        """
        Try to build rows one by one using greedy strategy with some randomization/shuffling?
        Actually, pure greedy might get stuck. Let's use a backtracking with greedy heuristic.
        """
        used_diffs = set()
        rows = []
        
        # To increase diversity, we can shuffle the order in which we try to fill rows?
        # Or just try to fill row 0, then 1, etc.
        # If it fails, we might need to backtrack.
        # For now, let's try a simple backtracking that uses greedy as a first choice.
        
        def backtrack(row_idx, last_val_constraint):
            if row_idx == n_rows:
                return True
            
            # Try to build a row
            # We can try different starting strategies.
            # For now, let's just use greedy_row but if it fails, we might need to backtrack.
            # However, greedy_row is deterministic given used_diffs. 
            # To allow backtracking, we need to try alternatives.
            
            # Let's implement a simple recursive search for the current row
            # that respects the constraint and tries to keep values small.
            
            def build_row(pos, current_row, current_max):
                if pos == num_elements:
                    if current_row[-1] < last_val_constraint:
                        return False
                    
                    new_diffs = set()
                    for i in range(len(current_row)):
                        for j in range(i+1, len(current_row)):
                            d = current_row[j] - current_row[i]
                            if d in used_diffs:
                                return False
                            new_diffs.add(d)
                            
                    rows.append(current_row)
                    used_diffs.update(new_diffs)
                    
                    if backtrack(row_idx + 1, current_row[-1]):
                        return True
                    
                    rows.pop()
                    used_diffs.difference_update(new_diffs)
                    return False
                
                min_val = current_row[-1] + 1 if current_row else 1
                # Heuristic: Try to keep values small, but allow up to max_scope
                # Also, we must ensure enough room for remaining elements
                max_val = max_scope - (num_elements - pos - 1)
                
                if min_val > max_val:
                    return False
                
                # Optimization: Check validity of diff before recursing
                for val in range(min_val, max_val + 1):
                    # Check if val - x for all x in current_row are not in used_diffs
                    # AND are distinct from each other (they are, since x are distinct)
                    valid = True
                    for x in current_row:
                        d = val - x
                        if d in used_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    if build_row(pos + 1, current_row + [val], val):
                        return True
                
                return False
            
            return build_row(1, [0], 0)
        
        if backtrack(0, 0):
            return rows
        return None

    # Strategy:
    # 1. Try to find a solution with scope 105, 106, ..., up to 112.
    # 2. The backtracking above is the main engine.
    # 3. To speed it up, we can use randomization in the order of value selection if we get stuck,
    #    but since we are iterating scope, we want the first valid solution.
    # 4. The current backtracking tries values in increasing order. This is good for minimizing scope.
    
    # However, the search space is huge. 
    # Let's add a time check inside the recursion to abort if we exceed a per-attempt time limit?
    # Or just rely on the global deadline.
    
    # Let's try scopes from 105 to 112.
    for M in range(105, 113):
        if time.time() > DEADLINE:
            break
            
        # Reset state for each M
        # We need to be careful with closures in Python if we redefine functions.
        # Let's wrap the solver in a class or use nonlocal carefully, or just define it inside.
        
        used_diffs = set()
        rows = []
        
        # We will use a global counter to abort if it takes too long for this M
        attempts = [0]
        
        def backtrack_row(row_idx, last_val_constraint):
            if time.time() > DEADLINE:
                return False
                
            if row_idx == n_rows:
                return True
            
            attempts[0] += 1
            if attempts[0] % 10000 == 0 and time.time() > DEADLINE:
                return False
            
            def build_row(pos, current_row, current_max):
                if time.time() > DEADLINE:
                    return False
                
                if pos == num_elements:
                    if current_row[-1] < last_val_constraint:
                        return False
                    
                    new_diffs = set()
                    for i in range(len(current_row)):
                        for j in range(i+1, len(current_row)):
                            d = current_row[j] - current_row[i]
                            if d in used_diffs:
                                return False
                            new_diffs.add(d)
                            
                    rows.append(current_row)
                    used_diffs.update(new_diffs)
                    
                    if backtrack_row(row_idx + 1, current_row[-1]):
                        return True
                    
                    rows.pop()
                    used_diffs.difference_update(new_diffs)
                    return False
                
                min_val = current_row[-1] + 1 if current_row else 1
                max_val = M - (num_elements - pos - 1)
                
                if min_val > max_val:
                    return False
                
                for val in range(min_val, max_val + 1):
                    valid = True
                    for x in current_row:
                        d = val - x
                        if d in used_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    if build_row(pos + 1, current_row + [val], val):
                        return True
                
                return False
            
            return build_row(1, [0], 0)
        
        if backtrack_row(0, 0):
            return rows

    # If we didn't find a solution with scope <= 112 in time, 
    # fall back to a greedy solution with a larger scope to ensure validity.
    # The greedy solution might have a larger scope but is valid.
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < num_elements:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows
