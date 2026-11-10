import time
import sys

def construct(n_rows, k):
    deadline = time.time() + 145  # 5 second safety margin
    
    def greedy_solve():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                x += 1
            rows.append(row)
        return rows, max(max(r) for r in rows)
    
    best_rows, best_scope = greedy_solve()
    
    if time.time() > deadline:
        return best_rows
    
    # Backtracking with MRV to try to improve
    # We'll try to construct rows with a tighter scope
    
    def backtrack(rows, used, scope_limit, row_idx):
        nonlocal best_rows, best_scope
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            # Found a valid solution
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
            return True
        
        # Build the next row with MRV
        row = [0]
        used_before = set(used)
        
        def build_row(row, used, pos):
            nonlocal best_rows, best_scope
            if time.time() > deadline:
                return False
            
            if pos == k + 1:
                # Row complete, proceed to next row
                return backtrack(rows + [row], used, scope_limit, row_idx + 1)
            
            last = row[-1]
            # Try candidate values
            for x in range(last + 1, scope_limit + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                # MRV check: count how many future positions are still feasible
                # Simple heuristic: prefer smaller x to keep scope tight
                row.append(x)
                if build_row(row, used | new_diffs, pos + 1):
                    return True
                row.pop()
            
            return False
        
        if build_row(row, used, 1):
            return True
        
        return False
    
    # Try to improve with backtracking, starting from a tighter scope
    # First, let's try to find a solution with scope = best_scope - 1, then -2, etc.
    # But this might be too slow. Instead, let's do a limited backtracking search.
    
    # Alternative: Use a different greedy strategy with randomization and restarts
    import random
    
    def randomized_greedy(seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                # Try candidates in a randomized order but biased toward small values
                candidates = []
                for x_try in range(x, x + 50):  # Look ahead 50
                    new_diffs = {x_try - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x_try)
                if not candidates:
                    x += 1
                    continue
                # Pick the smallest to keep scope tight, or randomize
                if rng.random() < 0.7:
                    x = candidates[0]
                else:
                    x = rng.choice(candidates[:min(5, len(candidates))])
                new_diffs = {x - a for a in row}
                row.append(x)
                used |= new_diffs
                x += 1
            rows.append(row)
        return rows, max(max(r) for r in rows)
    
    # Try multiple randomized greedy attempts
    for seed in range(20):
        if time.time() > deadline:
            break
        rows, scope = randomized_greedy(seed)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
            # Try to improve further from this point
            if time.time() < deadline - 10:
                # Do a local search or limited backtracking
                pass
    
    return best_rows
