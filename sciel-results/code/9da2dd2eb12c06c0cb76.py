import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Each row has k+1 elements, C(k+1, 2) differences
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Greedy fallback
    def greedy_construct():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Get initial greedy solution
    best_result = greedy_construct()
    best_scope = max(max(r) for r in best_result)
    
    # More sophisticated search: build rows one at a time with backtracking
    # but use smarter pruning
    
    def find_solution(scope_limit, max_time=10.0):
        """Try to find a valid construction with max entry <= scope_limit"""
        local_start = time.time()
        used_diffs = set()
        rows = []
        
        def build_row():
            nonlocal used_diffs
            if time.time() - local_start > max_time or time.time() > deadline:
                return None
            
            row = [0]
            row_diffs = set()
            
            def extend_row(pos, last_val):
                nonlocal used_diffs
                if time.time() - local_start > max_time or time.time() > deadline:
                    return None
                
                if pos == k + 1:
                    # Check all diffs are valid
                    all_diffs = set()
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            if d in used_diffs or d in row_diffs:
                                return None
                            all_diffs.add(d)
                    # Valid row
                    return row.copy()
                
                # Try next values
                for x in range(last_val + 1, scope_limit + 1):
                    # Quick check: differences to existing elements
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs or d in row_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if not valid:
                        continue
                    
                    # Check distinctness within row
                    if len(new_diffs) != len(row):
                        continue
                    
                    row.append(x)
                    row_diffs.update(new_diffs)
                    
                    result = extend_row(pos + 1, x)
                    if result is not None:
                        return result
                    
                    row.pop()
                    row_diffs.difference_update(new_diffs)
                
                return None
            
            return extend_row(1, 0)
        
        def construct_all():
            nonlocal used_diffs, rows
            if len(rows) == n_rows:
                return True
            
            if time.time() - local_start > max_time or time.time() > deadline:
                return False
            
            row = build_row()
            if row is None:
                return False
            
            # Add all diffs from this row to used
            row_all_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    row_all_diffs.add(row[j] - row[i])
            
            # Check no conflicts
            if row_all_diffs & used_diffs:
                return False
            
            rows.append(row)
            used_diffs.update(row_all_diffs)
            
            if construct_all():
                return True
            
            # Backtrack
            rows.pop()
            used_diffs.difference_update(row_all_diffs)
            return False
        
        if construct_all():
            return rows
        return None
    
    # Try to find better solutions by reducing scope
    # Start from best_scope - 1 and go down
    for scope in range(best_scope - 1, total_diffs - 1, -1):
        if time.time() > deadline:
            break
        
        # Try a few random seeds for diversity
        for seed in range(20):
            if time.time() > deadline:
                break
            
            random.seed(seed * 1000 + scope)
            
            # Shuffle the order of trying values slightly for diversity
            result = find_solution(scope, max_time=5.0)
            if result is not None:
                new_scope = max(max(r) for r in result)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_result = result
                break  # Found a solution at this scope, move to next
    
    return best_result
