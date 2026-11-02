import time
import random
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_rows = None
    best_scope = float('inf')
    
    def get_scope(rows):
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def greedy_row(used, k, randomize=False, cost_fn=None):
        """Build one row greedily."""
        row = [0]
        while len(row) < k + 1:
            candidates = []
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    if cost_fn:
                        cost = cost_fn(x, new_diffs, used, row, k)
                    else:
                        cost = 0
                    candidates.append((cost, x))
                    if not randomize:
                        break
                    # For randomize, collect a few candidates
                    if len(candidates) >= 20:
                        break
                x += 1
                if x > 200:  # safety
                    break
            
            if not candidates:
                return None
            
            if randomize:
                # Pick randomly from top candidates
                candidates.sort(key=lambda t: t[0])
                # Pick from top 5
                top = candidates[:min(5, len(candidates))]
                _, x = random.choice(top)
            else:
                _, x = candidates[0]
            
            row.append(x)
            used.update({x - a for a in row[:-1]})
        
        return row
    
    # Strategy 1: Standard greedy (deterministic)
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            if random.random() < 0.3:
                row = greedy_row(used, k, randomize=True)
            else:
                row = greedy_row(used, k, randomize=False)
            if row is None:
                success = False
                break
            rows.append(row)
        
        if success and is_valid(rows):
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback: pure greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
                        break
                    x += 1
            rows.append(row)
        return rows
    
    # Local improvement
    def local_improve(rows, used_set):
        """Try to reduce the scope by adjusting marks."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            scope = get_scope(rows)
            # Try to reduce the largest mark in each row
            for r_idx in range(len(rows)):
                row = rows[r_idx]
                # Find the largest mark
                max_idx = max(range(len(row)), key=lambda i: row[i])
                max_val = row[max_idx]
                # Try to reduce it
                for new_val in range(max_val - 1, row[max_idx-1], -1):
                    # Check if new_val is valid
                    new_diffs = {new_val - a for i, a in enumerate(row) if i != max_idx}
                    # Also need to check differences between other marks remain valid
                    # Actually, we only change one mark, so other differences stay the same
                    # But we need to remove old differences involving max_val and add new ones
                    old_diffs = {max_val - a for a in row if a != max_val}
                    # Also differences a - max_val for a > max_val, but max_val is largest so none
                    # Check new_diffs don't conflict
                    if not new_diffs & (used_set - old_diffs):
                        # Also check new_val > row[max_idx-1] and new_val < row[max_idx+1] if exists
                        if max_idx > 0 and new_val <= row[max_idx-1]:
                            continue
                        if max_idx < len(row)-1 and new_val >= row[max_idx+1]:
                            continue
                        # Apply the change
                        used_set -= old_diffs
                        used_set |= new_diffs
                        row[max_idx] = new_val
                        improved = True
                        break
        
        return rows, used_set
    
    # Rebuild used_set
    used_set = set()
    for row in best_rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used_set.add(row[j] - row[i])
    
    best_rows, used_set = local_improve(best_rows, used_set)
    
    return best_rows
