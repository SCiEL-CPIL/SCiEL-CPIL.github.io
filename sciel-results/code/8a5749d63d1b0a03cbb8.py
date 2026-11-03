import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Greedy Initialization
    def greedy_construct(n, k):
        used = set()
        rows = []
        for _ in range(n):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows, used

    rows, used = greedy_construct(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in best_rows)
    
    def get_max_val(rows):
        return max(max(r) for r in rows)

    # Backtracking search for a single row
    # used_diffs: set of forbidden differences
    # k: number of intervals (so k+1 marks)
    # max_limit: maximum value for the last mark
    # prefer_smaller: if True, try smaller x first. If False, randomize.
    def find_row(used_diffs, k, max_limit, randomize=False):
        row = [0]
        current_diffs = set()
        
        def backtrack(idx, last_val):
            if idx == k: # We have k+1 marks (0 to k)
                return row[:]
            
            # Determine range for next value
            start = last_val + 1
            end = max_limit
            
            if randomize:
                candidates = list(range(start, end + 1))
                random.shuffle(candidates)
            else:
                candidates = range(start, end + 1)
                
            for x in candidates:
                new_diffs = {x - a for a in row}
                if new_diffs & used_diffs:
                    continue
                # Check internal consistency is inherent since row is increasing
                row.append(x)
                current_diffs.update(new_diffs)
                
                res = backtrack(idx + 1, x)
                if res is not None:
                    return res
                    
                row.pop()
                current_diffs.difference_update(new_diffs)
                
            return None
        
        return backtrack(1, 0)

    # Main optimization loop
    while time.time() < deadline:
        # Find the row with the largest maximum value
        max_val = -1
        max_idx = -1
        for i, r in enumerate(rows):
            if r[-1] > max_val:
                max_val = r[-1]
                max_idx = i
        
        # Remove the worst row's differences
        worst_row = rows[max_idx]
        worst_diffs = set()
        for i in range(len(worst_row)):
            for j in range(i+1, len(worst_row)):
                worst_diffs.add(worst_row[j] - worst_row[i])
        
        used.difference_update(worst_diffs)
        
        # Try to find a better row (smaller max)
        found_better = False
        target_max = max_val - 1
        if target_max >= k: # Minimal possible max for k+1 marks is k
            new_row = find_row(used, k, target_max, randomize=False)
            if new_row is not None:
                rows[max_idx] = new_row
                used.update({new_row[j] - new_row[i] for i in range(len(new_row)) for j in range(i+1, len(new_row))})
                found_better = True
                current_max = get_max_val(rows)
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in rows]
        
        if not found_better:
            # Try to find an alternative row with the same max (to change the landscape)
            # Use randomization to find a different valid row
            new_row = find_row(used, k, max_val, randomize=True)
            if new_row is not None and new_row != worst_row:
                rows[max_idx] = new_row
                used.update({new_row[j] - new_row[i] for i in range(len(new_row)) for j in range(i+1, len(new_row))})
            else:
                # Restore original
                rows[max_idx] = worst_row
                used.update(worst_diffs)
                
        # Occasionally, if we are stuck, try a random restart or shuffle
        # This is a heuristic to escape local minima where no single row improvement is possible
        # but a multi-row change is.
        # For now, the "alternative row" step provides some exploration.
        # If best_max hasn't improved in a long time, we might need a stronger move.
        # But given the small size (7 rows, 6 marks), the search space is manageable.

    return best_rows
