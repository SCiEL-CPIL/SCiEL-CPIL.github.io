import sys
sys.setrecursionlimit(10000)

_RULER_CACHE = {}

def _generate_rulers(max_mark, num_marks):
    key = (max_mark, num_marks)
    if key in _RULER_CACHE:
        return _RULER_CACHE[key]
    
    rulers = []
    current_ruler = [0]
    current_diffs = set()
    
    def backtrack(last_mark):
        if len(current_ruler) == num_marks:
            rulers.append((tuple(current_ruler), frozenset(current_diffs), current_ruler[-1]))
            return
        
        # Heuristic: If we are running out of space to place the remaining marks
        # such that differences are distinct, we can prune.
        # But simple bound: next_mark <= max_mark
        
        for next_mark in range(last_mark + 1, max_mark + 1):
            new_diffs = set()
            valid = True
            for prev in current_ruler:
                d = next_mark - prev
                if d in current_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            current_ruler.append(next_mark)
            current_diffs.update(new_diffs)
            
            backtrack(next_mark)
            
            current_ruler.pop()
            current_diffs.difference_update(new_diffs)
            
    backtrack(0)
    
    _RULER_CACHE[key] = rulers
    return rulers

def construct(n_rows, k):
    num_marks = k + 1
    total_diffs = n_rows * (num_marks * (num_marks - 1) // 2)
    min_bound = total_diffs
    
    # We know the answer is around 112 for 7,5.
    # Set a reasonable upper bound to avoid generating too many rulers.
    # If min_bound is small, we still need to go up.
    upper_bound = max(min_bound + 50, 150)
    
    all_rulers = _generate_rulers(upper_bound, num_marks)
    all_rulers.sort(key=lambda x: x[2])
    
    for B in range(min_bound, upper_bound + 1):
        candidates = []
        for r in all_rulers:
            if r[2] <= B:
                candidates.append(r)
            else:
                break
        
        if len(candidates) < n_rows:
            continue
            
        def dfs(idx, used_diffs, rows):
            if len(rows) == n_rows:
                return rows
            
            if idx >= len(candidates):
                return None
            
            if len(candidates) - idx < n_rows - len(rows):
                return None
            
            for i in range(idx, len(candidates)):
                ruler_tuple, diff_set, max_mark_val = candidates[i]
                
                if diff_set & used_diffs:
                    continue
                
                new_used = used_diffs | diff_set
                new_rows = rows + [list(ruler_tuple)]
                
                res = dfs(i + 1, new_used, new_rows)
                if res is not None:
                    return res
                
            return None
        
        solution = dfs(0, set(), [])
        
        if solution is not None:
            return solution
            
    # Fallback
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < num_marks:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
