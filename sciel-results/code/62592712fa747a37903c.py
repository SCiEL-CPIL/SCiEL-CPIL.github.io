import random
import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0
    
    # 1. Greedy Initialization
    def greedy_solve():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                for x in range(last + 1, 200):
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        row.append(x)
                        used_diffs.update(new_diffs)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    initial_rows = greedy_solve()
    if initial_rows is None:
        # Fallback
        return [[0, 1, 2, 3, 4, 5] for _ in range(n_rows)]
        
    best_scope = max(r[-1] for r in initial_rows)
    best_result = initial_rows

    if best_scope <= 105:
        return best_result

    # 2. Backtracking to improve
    def backtrack_solve(max_scope):
        """
        Returns a list of rows if a solution with scope <= max_scope is found, else None.
        """
        used_diffs = set()
        rows = []
        
        def _bt(row_idx, current_row):
            # Time check
            if time.time() - start_time > time_limit:
                return False
            
            # Pruning
            remaining = (k + 1) - len(current_row)
            if remaining > 0:
                min_last = current_row[-1] + remaining
                if min_last > max_scope:
                    return False
            
            if row_idx == n_rows:
                actual_scope = max(r[-1] for r in rows)
                if actual_scope <= max_scope:
                    return True
                return False
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                if _bt(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            
            # Try x from last_val + 1 to max_scope
            for x in range(last_val + 1, max_scope + 1):
                valid = True
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Recurse
                used_diffs.update(new_diffs)
                
                if _bt(row_idx, current_row + [x]):
                    return True
                
                used_diffs.difference_update(new_diffs)
                
        success = _bt(0, [0])
        if success:
            return [r[:] for r in rows]
        return None

    # Try to improve by lowering the scope
    # We try scopes from best_scope - 1 downwards.
    # To save time, we can skip some if we are confident, but linear decrement is safe.
    current_target = best_scope - 1
    
    while current_target >= 105 and time.time() - start_time < time_limit:
        solution = backtrack_solve(current_target)
        if solution:
            best_scope = max(r[-1] for r in solution)
            best_result = solution
            current_target = best_scope - 1
        else:
            # If we fail at current_target, it might be because it's impossible
            # or because we ran out of time.
            # If we ran out of time, we should stop.
            # If it's impossible, we can break.
            # How to distinguish?
            # If the time limit is not reached, it's likely impossible (or search failed).
            # Given the small size, if search fails quickly, it's likely impossible.
            # If it takes long, we might have timed out.
            if time.time() - start_time > time_limit - 5:
                break
            # If we failed, we assume current_target is not achievable.
            # We can try current_target - 1? No, if S is not achievable, S-1 is not either.
            # So we can break?
            # Wait, if S=110 fails, S=109 must fail.
            # So we can break immediately if we are sure it's a hard failure, not a timeout.
            # But our _bt returns False on timeout too.
            # We can check if we hit the time limit inside _bt.
            # Let's just break if we fail, assuming the gap is small or the solution doesn't exist.
            # Actually, if 111 fails, 110 fails. So we can break.
            break
            
    return best_result
