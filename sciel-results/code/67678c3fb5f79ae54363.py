import sys
import time
import random

def construct(n_rows, k):
        # We want to minimize the scope (max entry).
        # We will use a backtracking search.
        
        # To handle the time limit, we might need to use a heuristic search
        # or a very pruned DFS.
        
        # Given n=4, k=8, the search space is manageable with good pruning.
        
        best_solution = None
        best_scope = float('inf')
        
        # Precompute a tight upper bound?
        # Counting bound is 112. Known best is 134.
        # We can try to find a solution with scope < 134.
        
        # Let's try to construct rows one by one.
        # We will use a recursive function.
        
        used_diffs = set()
        rows = []
        
        # To improve speed, we can fix the first row to a known good Golomb ruler
        # or just let the search find it.
        # Finding a Golomb ruler of order 9 is itself a search problem.
        # But since we need 4 rows, the first row doesn't have to be optimal,
        # just valid.
        
        # Heuristic: Try to keep numbers small.
        
        def solve(row_idx, current_row, current_max):
            nonlocal best_solution, best_scope
            
            # Pruning: if current_max > best_scope, no point continuing
            if current_max > best_scope:
                return False
            
            if row_idx == n_rows:
                # Found a valid solution
                if current_max < best_scope:
                    best_scope = current_max
                    best_solution = [list(r) for r in rows]
                return True
            
            # If current_row is complete, move to next row
            if len(current_row) == k + 1:
                rows.append(current_row)
                if solve(row_idx + 1, [], current_max):
                    return True
                rows.pop()
                return False
            
            # Determine next index
            i = len(current_row)
            
            # Start value
            start_val = current_row[-1] + 1 if current_row else 1
            
            # Upper bound for this value?
            # We don't know the final scope yet, but we know it must be <= best_scope.
            # So v <= best_scope.
            
            # Optimization:
            # If we are in the last row, we can be more aggressive with pruning.
            
            # Try values
            # To make it faster, we might want to try values that are "sparse" or "dense"?
            # Usually, for Golomb rulers, greedy (smallest possible) works well for the first row.
            
            # However, simple DFS might be too slow if the first row is not constrained enough.
            # Let's add a heuristic: 
            # If row_idx == 0, we can try to construct a "standard" Golomb ruler.
            # But for now, let's stick to DFS with time check.
            
            # To avoid infinite loops or slow searches, we can limit the search depth or time.
            # But we are in a recursive function.
            
            # We need to check differences.
            # For a candidate v, the new diffs are {v - x for x in current_row}.
            # If any diff is in used_diffs, reject.
            
            # Also, we should check if the set of new diffs is internally consistent?
            # No, diffs within a row are always distinct if the row is strictly increasing?
            # Wait. In a row 0, 1, 3. Diffs: 1-0=1, 3-1=2, 3-0=3. Distinct.
            # In a row 0, 2, 3. Diffs: 2, 1, 3. Distinct.
            # Yes, for any strictly increasing sequence, pairwise differences are distinct?
            # NO.
            # Example: 0, 1, 2, 4.
            # Diffs: 1, 2, 4, 1, 3, 2.
            # 1-0=1, 2-1=1. Conflict!
            # So a row must be a Golomb Ruler itself.
            # So we must ensure that the new value v does not create a duplicate difference *within the current row*.
            
            # So check 1: new_diffs are distinct among themselves.
            # Check 2: new_diffs are not in used_diffs (global).
            
            # Let's implement this.
            
            for v in range(start_val, best_scope + 1):
                new_diffs = []
                conflict_internal = False
                for x in current_row:
                    d = v - x
                    if d in new_diffs:
                        conflict_internal = True
                        break
                    new_diffs.append(d)
                
                if conflict_internal:
                    continue
                
                # Check global conflicts
                conflict_global = False
                for d in new_diffs:
                    if d in used_diffs:
                        conflict_global = True
                        break
                
                if conflict_global:
                    continue
                
                # Accept v
                current_row.append(v)
                for d in new_diffs:
                    used_diffs.add(d)
                
                new_max = max(current_max, v)
                
                if solve(row_idx, current_row, new_max):
                    return True
                
                # Backtrack
                for d in new_diffs:
                    used_diffs.remove(d)
                current_row.pop()
            
            return False
        
        # Start the search
        # To ensure we finish in time, we might need to randomize or use a heuristic start.
        # Pure DFS might get stuck in a bad branch for the first row.
        # Let's try to find a valid first row quickly using a greedy method, then backtrack for the rest.
        
        # Greedy for first row
        def greedy_first_row(k):
            row = [0]
            local_diffs = set()
            for _ in range(k):
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & local_diffs):
                        row.append(x)
                        local_diffs |= new_diffs
                        break
                    x += 1
            return row

        # Initialize with a greedy row to have a valid starting point?
        # No, the DFS above handles it. But DFS might be slow.
        # Let's add a time limit check in the recursion.
        
        start_time = time.time()
        
        # To improve the chance of finding a good solution quickly,
        # we can try different orderings or heuristics.
        
        # Let's run the DFS.
        # If it takes too long, we might need a fallback.
        
        # To make the DFS faster, we can use a heuristic for the first row:
        # Try to make the first row "dense" or "sparse"?
        # Actually, the first row consumes a lot of small differences.
        # It is often better to have the first row be a "good" Golomb ruler.
        
        # Let's try a few random restarts if the first attempt is slow?
        # Or just run the DFS with a tight time limit per node?
        
        # Given the constraints (120s), and the small size, DFS should work.
        # But to be safe, we'll add a global time check.
        
        def solve_with_time(row_idx, current_row, current_max):
            nonlocal best_solution, best_scope
            
            if time.time() - start_time > 110:
                return False
                
            if current_max > best_scope:
                return False
            
            if row_idx == n_rows:
                if current_max < best_scope:
                    best_scope = current_max
                    best_solution = [list(r) for r in rows]
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                if solve_with_time(row_idx + 1, [], current_max):
                    return True
                rows.pop()
                return False
            
            i = len(current_row)
            start_val = current_row[-1] + 1 if current_row else 1
            
            # Heuristic: 
            # If we are in the first row, we might want to try values that leave room for others.
            # For now, try in increasing order.
            
            # Optimization: 
            # If we have a best_scope, we can limit v to best_scope.
            
            for v in range(start_val, best_scope + 1):
                if time.time() - start_time > 110:
                    return False
                    
                new_diffs = []
                conflict_internal = False
                for x in current_row:
                    d = v - x
                    if d in new_diffs:
                        conflict_internal = True
                        break
                    new_diffs.append(d)
                
                if conflict_internal:
                    continue
                
                conflict_global = False
                for d in new_diffs:
                    if d in used_diffs:
                        conflict_global = True
                        break
                
                if conflict_global:
                    continue
                
                current_row.append(v)
                for d in new_diffs:
                    used_diffs.add(d)
                
                new_max = max(current_max, v)
                
                if solve_with_time(row_idx, current_row, new_max):
                    return True
                
                for d in new_diffs:
                    used_diffs.remove(d)
                current_row.pop()
            
            return False

        # Initial best_scope from greedy?
        # Let's get a greedy solution first to set an upper bound.
        def greedy_solution(n_rows, k):
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

        init_rows = greedy_solution(n_rows, k)
        best_scope = max(max(r) for r in init_rows)
        best_solution = init_rows
        
        # Now run DFS to improve
        # We need to clear used_diffs and rows for the DFS
        used_diffs.clear()
        rows.clear()
        
        # The DFS above uses `used_diffs` and `rows` from the enclosing scope.
        # We need to make sure they are accessible.
        
        # Note: The DFS might be very slow if it explores many bad branches.
        # To mitigate, we can use a "randomized" DFS or try specific values first.
        # But for now, let's try the standard DFS.
        
        # To ensure we return *something* valid even if DFS times out,
        # we rely on `best_solution` being updated only when a complete valid row set is found.
        # Since `init_rows` is valid, `best_solution` is always valid.
        
        # Let's run the DFS.
        solve_with_time(0, [], 0)
        
        return best_solution
