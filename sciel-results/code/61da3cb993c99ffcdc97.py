import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # safety margin under 150s kill
    
    def time_left():
        return deadline - time.time()
    
    def row_diffs_set(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs
    
    def all_diffs(rows):
        s = set()
        for r in rows:
            s |= row_diffs_set(r)
        return s
    
    def scope(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        seen = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True
    
    # Backtracking row construction with heuristic ordering
    def backtrack_row(used_diffs, max_val_hint, jitter=0.0, seed=None):
        """Construct a single row of k+1 increasing integers starting at 0,
        avoiding all differences in used_diffs. Returns row or None."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        row = [0]
        row_diffs = set()
        
        def solve(pos, current_max):
            if pos == k:
                return row[:]
            
            # Try candidate values in order, with optional jitter
            start = row[-1] + 1
            # Heuristic: try smaller values first to minimize scope
            # But with jitter, sometimes skip to explore
            candidates = list(range(start, max_val_hint + 1))
            if jitter > 0:
                # Randomly shuffle a portion to escape local patterns
                shuffle_start = min(len(candidates), int(len(candidates) * jitter))
                for i in range(shuffle_start, len(candidates) - 1):
                    j = rng.randint(i, len(candidates) - 1)
                    candidates[i], candidates[j] = candidates[j], candidates[i]
            
            for x in candidates:
                new_diffs = {x - a for a in row}
                if new_diffs & used_diffs:
                    continue
                if new_diffs & row_diffs:
                    continue
                # Check if this leads to a dead end quickly (light lookahead)
                row.append(x)
                row_diffs.update(new_diffs)
                
                result = solve(pos + 1, x)
                if result is not None:
                    return result
                
                row.pop()
                row_diffs -= new_diffs
            
            return None
        
        result = solve(1, 0)
        return result
    
    # Greedy baseline
    def greedy_construct():
        used = set()
        rows = []
        for r_idx in range(n_rows):
            row = [0]
            row_diffs = set()
            for _ in range(k):
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and not (new_diffs & row_diffs):
                        break
                    x += 1
                row.append(x)
                row_diffs |= new_diffs
                used |= new_diffs
            rows.append(row)
        return rows
    
    current_rows = greedy_construct()
    best_rows = [r[:] for r in current_rows]
    best_scope = scope(current_rows)
    
    # Per-row local optimization: try to rebuild a row with smaller max
    def optimize_row(rows, row_idx, max_attempts=50):
        """Try to find a better (smaller max) row at row_idx."""
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max
        
        # Collect all diffs from other rows
        other_diffs = set()
        for j in range(n_rows):
            if j != row_idx:
                other_diffs |= row_diffs_set(rows[j])
        
        for attempt in range(max_attempts):
            if time_left() < 3:
                break
            
            # Use backtracking with increasing max_val_hint
            # Start with current_max - 1, expand if needed
            hint = current_max - 1 - attempt
            if hint < k:  # minimum possible
                break
            
            # Add jitter to escape local minima
            jitter = 0.05 + 0.1 * (attempt / max_attempts)
            seed = random.randint(0, 1000000)
            
            new_row = backtrack_row(other_diffs, hint, jitter=jitter, seed=seed)
            if new_row is not None:
                new_max = max(new_row)
                if new_max < best_max:
                    # Verify full validity
                    test_rows = [list(r) for r in rows]
                    test_rows[row_idx] = new_row
                    if is_valid(test_rows):
                        best_row = list(new_row)
                        best_max = new_max
        
        return best_row, best_max
    
    # Local polish: iterate over rows, trying to reduce each
    def local_polish(rows, max_passes=8):
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        while improved and time_left() > 10 and pass_count < max_passes:
            improved = False
            pass_count += 1
            # Process rows in order of largest max first
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 10:
                    break
                new_row, new_max = optimize_row(rows, i, max_attempts=30)
                if new_max < max(rows[i]):
                    rows[i] = new_row
                    if scope(rows) < best_scope:
                        nonlocal best_scope, best_rows
                        best_scope = scope(rows)
                        best_rows = [r[:] for r in rows]
                    improved = True
        return rows
    
    # ILS: perturb by rebuilding random rows, then polish
    def kick(rows):
        """Perturb by rebuilding 1-2 random rows with high randomness."""
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 1, 2])
        indices = random.sample(range(n_rows), min(num_kicks, n_rows))
        
        for idx in indices:
            other_diffs = set()
            for j in range(n_rows):
                if j != idx:
                    other_diffs |= row_diffs_set(rows[j])
            
            # High jitter to explore far from current
            hint = max(rows[idx]) + random.randint(0, 20)
            jitter = 0.3 + 0.3 * random.random()
            seed = random.randint(0, 1000000)
            
            new_row = backtrack_row(other_diffs, hint, jitter=jitter, seed=seed)
            if new_row is not None:
                # Verify validity
                test_rows = [list(r) for r in rows]
                test_rows[idx] = new_row
                if is_valid(test_rows):
                    rows[idx] = new_row
        
        return rows
    
    # Main search loop
    no_improve_count = 0
    max_no_improve = 20
    
    while time_left() > 15:
        # Local polish
        polished = local_polish(current_rows, max_passes=4)
        current_rows = polished
        
        if scope(current_rows) < best_scope:
            best_scope = scope(current_rows)
            best_rows = [r[:] for r in current_rows]
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If stuck, kick
        if no_improve_count >= max_no_improve or random.random() < 0.1:
            kicked = kick(current_rows)
            if scope(kicked) < best_scope + 30:  # don't accept too-bad kicks
                current_rows = kicked
            no_improve_count = 0
    
    return best_rows
