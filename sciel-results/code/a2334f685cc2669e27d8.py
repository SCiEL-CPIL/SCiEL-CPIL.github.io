import time
import random

def _construct_row(forbidden_diffs, k, max_val_limit=None):
    """
    Construct a single row of length k+1 starting with 0.
    Uses backtracking to find a row with minimal last element.
    Returns the row as a list, or None if not found within max_val_limit.
    """
    row = [0]
    
    def backtrack(idx, current_max):
        if idx == k + 1:
            return current_max
        
        # Pruning: if we have a limit, don't go over it
        if max_val_limit is not None and current_max > max_val_limit:
            return None
            
        # Try next values
        # We can randomize the order of trying values to escape local minima in the outer loop
        # But for finding *a* row, sequential is fine. For optimization, we want the smallest.
        # Let's try values from current_max + 1 upwards.
        
        # To speed up, we can estimate the minimum possible next value.
        # The next value x must satisfy x - a not in forbidden for all a in row.
        
        # Heuristic: try small increments first
        start_val = row[-1] + 1
        # We can limit the search depth if we just want *a* solution, 
        # but here we want the *best* row given the forbidden set.
        
        # However, finding the absolute minimum row is NP-hard in general, 
        # but for k=5 it's small.
        
        # Let's try a range of values.
        # To avoid infinite loops, we need a cap.
        # The global best scope is 112. We can cap the search for a single row at, say, 150.
        
        found = None
        # Randomize the starting point slightly? No, let's stick to deterministic for row construction 
        # but randomize which row we modify in the outer loop.
        
        # Actually, a simple greedy for the row is:
        # x = row[-1] + 1
        # while True:
        #   if all(x - a not in forbidden_diffs for a in row):
        #      break
        #   x += 1
        # This finds the lexicographically smallest row.
        # Is it optimal for minimizing the last element?
        # Not necessarily. Sometimes a larger intermediate element allows a smaller final element.
        # Example: row [0, 2, 3] vs [0, 1, 4]. 
        # If we pick 1, we might be forced to pick 5 later. If we pick 2, we might pick 4.
        # So backtracking is better.
        
        # Given the small size, let's do a DFS with pruning.
        # State: current row, next index.
        # To keep it fast, we limit the value of the next element.
        
        # Let's try values from row[-1] + 1 to row[-1] + 50 (heuristic cap)
        # If we find a valid completion, return it.
        # To find the *best*, we'd need to explore all.
        # For the purpose of local search, finding a *good* row is enough.
        
        # Let's use a simple randomized greedy for the row construction to add diversity.
        # Or, just use the greedy smallest-first.
        
        pass

    # The above is getting complex. Let's simplify.
    # The main bottleneck is finding the set of 7 rows.
    # We will use a randomized greedy for the initial set, and then local search.
    
    # For the local search step, we replace one row.
    # We want to replace row i with a row that has a smaller max.
    # We can try to construct a row with max < current_max.
    
    # Let's implement a function to find *any* valid row with max < limit.
    def find_row_with_limit(forbidden, k, limit):
        row = [0]
        def bt(idx):
            if idx == k+1:
                return True
            # Try values from row[-1]+1 to limit
            # Randomize order?
            vals = list(range(row[-1]+1, limit+1))
            random.shuffle(vals)
            for v in vals:
                # Check conflicts
                ok = True
                for a in row:
                    if v - a in forbidden:
                        ok = False
                        break
                if ok:
                    row.append(v)
                    if bt(idx+1):
                        return True
                    row.pop()
            return False
        if bt(1):
            return row
        return None

    # In local search:
    # current_row_max = max(rows[i])
    # Try to find a row with max < current_row_max.
    # If found, accept.
    # If not, try to find a row with max < best_global_max.
    
    return None # Placeholder

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_rows = None
    best_score = float('inf')
    
    # Helper to check validity and get score
    def get_score(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        used = set()
        for r in rows:
            diffs = set()
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    if d in diffs:
                        return False
                    diffs.add(d)
            used.update(diffs)
        return True

    # Initial Generation: Randomized Greedy
    def generate_initial():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            # Randomize the step size or order?
            # Simple greedy:
            while len(row) < k + 1:
                x += 1
                # Add some randomness: sometimes jump ahead?
                # No, let's stick to greedy but restart if it gets too large.
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                else:
                    # If we are stuck, we can't easily backtrack in this simple loop.
                    # So if x gets too big (e.g. > 200), we fail this attempt.
                    if x > 200:
                        return None
            rows.append(row)
        return rows

    # Local Search
    def local_search(rows, time_limit):
        current_rows = [r[:] for r in rows]
        current_score = get_score(current_rows)
        used = set()
        for r in current_rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    used.add(r[j] - r[i])
        
        end_time = time.time() + time_limit
        while time.time() < end_time:
            # Pick a random row to modify
            idx = random.randint(0, n_rows - 1)
            old_row = current_rows[idx]
            old_max = max(old_row)
            
            # Remove old row's diffs from used
            old_diffs = set()
            for i in range(len(old_row)):
                for j in range(i+1, len(old_row)):
                    old_diffs.add(old_row[j] - old_row[i])
            for d in old_diffs:
                used.discard(d)
                
            # Try to find a better row
            # Target: max < old_max
            # If we can't, try max < current_score
            # Let's try to find a row with max < old_max
            # Use a randomized backtracking search for the new row
            
            new_row = None
            # We want to minimize the max.
            # Let's try to find *any* row with max < old_max.
            # If found, it's an improvement (or equal).
            # To ensure we don't loop, we can limit the search.
            
            # Simple approach: try to construct a row with greedy, but restrict max.
            # If greedy fails, try randomized backtracking.
            
            def try_find_row(limit):
                # Backtracking search for a row with max <= limit
                row = [0]
                def bt(idx):
                    if idx == k + 1:
                        return True
                    # Try values from row[-1]+1 to limit
                    # Randomize to find different solutions
                    start = row[-1] + 1
                    end = min(limit, 200) # Hard cap for speed
                    if start > end:
                        return False
                    vals = list(range(start, end + 1))
                    random.shuffle(vals)
                    for v in vals:
                        ok = True
                        for a in row:
                            if v - a in used:
                                ok = False
                                break
                        if ok:
                            row.append(v)
                            if bt(idx + 1):
                                return True
                            row.pop()
                    return False
                if bt(1):
                    return row[:]
                return None

            # Try to find a row with max < old_max
            if old_max > 1:
                new_row = try_find_row(old_max - 1)
            
            # If we found one, accept
            if new_row:
                new_max = max(new_row)
                # It is guaranteed to be < old_max
                # Update used
                for i in range(len(new_row)):
                    for j in range(i+1, len(new_row)):
                        used.add(new_row[j] - new_row[i])
                current_rows[idx] = new_row
                current_score = get_score(current_rows)
            else:
                # Try to find a row with max < current_score (if current_score < old_max)
                # Or just try to find any row to keep the state valid?
                # If we can't improve, we can try to swap with another row?
                # For now, if we can't improve this row, we just revert.
                for d in old_diffs:
                    used.add(d)
            
        return current_rows, current_score

    # Main Loop
    restarts = 0
    while time.time() < deadline and restarts < 50:
        rows = generate_initial()
        if rows:
            score = get_score(rows)
            if score < best_score:
                best_score = score
                best_rows = [r[:] for r in rows]
                
        # Local Search
        if best_rows:
            improved_rows, improved_score = local_search(best_rows, 10)
            if improved_score < best_score:
                best_score = improved_score
                best_rows = improved_rows
                
        # Also try local search on the initial random solution if it's good enough
        if rows and get_score(rows) < best_score + 10:
            ls_rows, ls_score = local_search(rows, 10)
            if ls_score < best_score:
                best_score = ls_score
                best_rows = ls_rows
                
        restarts += 1

    if best_rows is None:
        # Fallback to simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
        
    return best_rows
