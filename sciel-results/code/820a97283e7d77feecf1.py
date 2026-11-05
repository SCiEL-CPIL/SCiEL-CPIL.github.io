import random
import time

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    Uses backtracking with forward-checking heuristic: prefer candidates
    that leave the most valid options for subsequent positions.
    """
    start_time = time.time()
    time_limit = 110.0
    
    best_result = None
    best_scope = float('inf')
    
    def backtrack(row_idx, pos_in_row, current_row, used_diffs, scope_limit):
        """
        Backtracking search. current_row is a tuple.
        Returns (success, rows_list, final_scope) or (False, None, None).
        """
        if time.time() - start_time > time_limit:
            return (False, None, None)
        
        if pos_in_row > k:
            if row_idx + 1 == n_rows:
                final_scope = max(r[-1] for r in current_row)
                return (True, [list(r) for r in current_row], final_scope)
            else:
                return backtrack(row_idx + 1, 0, current_row + (0,), used_diffs, scope_limit)
        
        last_val = current_row[-1]
        candidates = []
        upper = scope_limit - 1 if scope_limit < float('inf') else 120
        
        for x in range(last_val + 1, upper + 1):
            new_diffs = {x - a for a in current_row}
            if new_diffs & used_diffs:
                continue
            
            # Heuristic: count valid next-position options (forward check)
            next_flex = 0
            if pos_in_row < k:
                new_row_with_x = current_row + (x,)
                for x2 in range(x + 1, min(x + 25, upper + 1)):
                    new_diffs2 = {x2 - a for a in new_row_with_x}
                    if not new_diffs2 & (used_diffs | new_diffs):
                        next_flex += 1
                        if next_flex >= 12:
                            break
            
            small_used = sum(1 for d in new_diffs if d <= 12)
            candidates.append((next_flex, small_used, x, new_diffs))
        
        if not candidates:
            return (False, None, None)
        
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
        
        top_n = min(8, len(candidates))
        order = list(range(top_n))
        if random.random() < 0.7:
            order = [0] + random.sample(order[1:], len(order) - 1)
        else:
            random.shuffle(order)
        
        for idx in order:
            _, _, x, new_diffs = candidates[idx]
            new_used = used_diffs | new_diffs
            new_row = current_row + (x,)
            
            success, rows, scope = backtrack(
                row_idx, pos_in_row + 1, new_row, new_used, scope_limit
            )
            if success:
                return (True, rows, scope)
        
        return (False, None, None)
    
    for scope in range(105, 125):
        if time.time() - start_time > time_limit:
            break
        
        for restart in range(150):
            if time.time() - start_time > time_limit:
                break
            
            random.seed(restart * 7919 + scope * 13)
            
            success, rows, final_scope = backtrack(
                0, 0, (0,), set(), scope
            )
            
            if success and final_scope < best_scope:
                best_scope = final_scope
                best_result = rows
                if best_scope <= 105:
                    break
        
        if best_scope <= 105:
            break
    
    if best_result is None:
        used = set()
        rows = []
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
    
    return best_result
