import time
import random

def construct(n_rows, k):
    deadline = time.time() + 110
    num_diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * num_diffs_per_row
    
    for M in range(total_diffs, total_diffs + 30):
        if time.time() > deadline:
            break
            
        # Try deterministic first
        result = _backtrack_solve(n_rows, k, M, deadline, node_limit=20000, randomize=False)
        if result is not None:
            return result
            
        # Try a few randomized restarts
        for attempt in range(5):
            if time.time() > deadline:
                break
            random.seed(M * 100 + attempt)
            result = _backtrack_solve(n_rows, k, M, deadline, node_limit=20000, randomize=True)
            if result is not None:
                return result
                
    return _greedy_construct(n_rows, k)

def _backtrack_solve(n_rows, k, M, deadline, node_limit=20000, randomize=False):
    used_diffs = set()
    rows = []
    nodes = [0]
    
    def backtrack(row_idx, pos, current_row):
        if time.time() > deadline:
            return False
        if nodes[0] > node_limit:
            return False
        nodes[0] += 1
        
        if row_idx == n_rows:
            return True
            
        if pos == k + 1:
            rows.append(current_row)
            success = backtrack(row_idx + 1, 0, [0])
            if success:
                return True
            rows.pop()
            return False
            
        last_val = current_row[-1]
        candidates = list(range(last_val + 1, M + 1))
        
        if randomize:
            random.shuffle(candidates)
        else:
            # For deterministic, we stick to increasing order
            pass
            
        for x in candidates:
            new_diffs = set()
            valid = True
            for r in current_row:
                d = x - r
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
                
            current_row.append(x)
            used_diffs.update(new_diffs)
            
            if backtrack(row_idx, pos + 1, current_row):
                return True
                
            current_row.pop()
            used_diffs.difference_update(new_diffs)
            
        return False
        
    return backtrack(0, 0, [0])

def _greedy_construct(n_rows, k):
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
