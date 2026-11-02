import random
import time

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows of k+1 integers.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # Pre-calculate all pairs (i, j) for difference checks
    # k+1 elements, indices 0..k
    # Differences are a[j] - a[i] for i < j
    
    def get_diffs(row):
        """Return a set of all positive differences in the row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_row(row, used_diffs):
        """Check if row is internally valid and disjoint from used_diffs."""
        row_diffs = get_diffs(row)
        if len(row_diffs) != len(row) * (len(row) - 1) // 2:
            return False
        if row_diffs & used_diffs:
            return False
        return True

    def backtrack_generate(used_diffs, rows, row_idx, max_val_bound):
        """
        Recursively generate rows using backtracking.
        Returns a list of rows if successful, else None.
        """
        if row_idx == n_rows:
            return rows
        
        # Randomize the order of candidate start positions? 
        # No, we build the row element by element.
        
        def build_row(current_row, used_diffs_local):
            if len(current_row) == k + 1:
                return current_row
            
            last_val = current_row[-1] if current_row else 0
            # We need to choose next_val > last_val
            # To keep max small, we should try smaller values first, but with some randomness.
            # Upper bound for next_val: 
            # If we want total max <= M, next_val <= M.
            # Also, we need enough room for remaining k+1-len(current_row) elements.
            
            # Heuristic: try values from last_val+1 upwards, but shuffle a window.
            # To avoid infinite loops, we cap the search.
            # A safe upper bound for a single element in a row of size k+1 with max M is M.
            # But we don't know M. Let's just scan up to a limit.
            
            # Optimization: We can prune if the remaining space is too small.
            
            candidates = []
            # Scan a window of candidates
            # For speed, we just scan sequentially but pick randomly from the first few valid ones?
            # Better: Scan all valid ones up to a limit, then pick one.
            # Limit: if we are at element i, the max possible value is roughly M.
            # Let's just scan up to 200 or so, as 112 is the target.
            
            x = last_val + 1
            # We must ensure we can fit the rest.
            # If we pick x, we need k+1-len(current_row)-1 more elements > x.
            # So x cannot be too close to the global max bound.
            # Since we don't have a fixed bound, we just scan.
            
            # To make it randomized, we collect candidates and pick one.
            # If we pick the smallest, it's greedy. If we pick randomly, it's diverse.
            
            # Let's scan up to last_val + 50 + random.randint(0, 50) to get diversity
            limit = last_val + 100 
            while x <= limit:
                new_diffs = {x - a for a in current_row}
                if not (new_diffs & used_diffs_local):
                    candidates.append(x)
                x += 1
            
            if not candidates:
                return None
            
            # Pick a candidate. 
            # Bias towards smaller values to keep max low, but allow some larger ones to escape dead ends.
            # Weights: inverse of value? Or just uniform?
            # Let's use a random choice from the first 10 valid candidates to keep it tight.
            if len(candidates) > 10:
                candidates = candidates[:10]
            
            chosen_x = random.choice(candidates)
            current_row.append(chosen_x)
            used_diffs_local.update({chosen_x - a for a in current_row[:-1]})
            
            result = build_row(current_row, used_diffs_local)
            if result is not None:
                return result
            else:
                # Backtrack
                current_row.pop()
                # Remove diffs
                for d in {chosen_x - a for a in current_row}:
                    used_diffs_local.discard(d)
                return None

        # Start building the row
        row = [0]
        # used_diffs for this row is initially empty, but we must check against global used_diffs
        # Wait, the recursive build_row needs to check against the GLOBAL used_diffs + diffs from current row.
        
        # Let's refactor: build_row takes the global used_diffs and the current row's partial diffs.
        
        def build_row_recursive(current_row, global_used, partial_diffs):
            if len(current_row) == k + 1:
                return current_row, global_used | partial_diffs
            
            last_val = current_row[-1]
            
            # Generate candidates
            candidates = []
            x = last_val + 1
            # Limit search to avoid explosion. 
            # If we are trying to find a solution with max <= 112, we can cap x at 112?
            # No, we want to minimize max. So we try small x first.
            # Let's cap x at 150 for safety.
            while x <= 150:
                new_diffs = {x - a for a in current_row}
                # Check against global used and partial diffs
                if not (new_diffs & (global_used | partial_diffs)):
                    candidates.append(x)
                x += 1
            
            if not candidates:
                return None, None
            
            # Randomize: pick from the first few valid candidates
            # To encourage smaller max, we prefer smaller x.
            # Let's pick randomly from the first 5 candidates.
            if len(candidates) > 5:
                candidates = candidates[:5]
            
            chosen_x = random.choice(candidates)
            new_partial = partial_diffs | {chosen_x - a for a in current_row}
            current_row.append(chosen_x)
            
            result, new_global = build_row_recursive(current_row, global_used, new_partial)
            if result is not None:
                return result, new_global
            else:
                current_row.pop()
                return None, None

        # Call recursive builder
        # Initial row is [0]
        # Global used is the `used_diffs` passed in
        # Partial diffs for [0] is empty
        new_row, new_used = build_row_recursive([0], used_diffs, set())
        
        if new_row is None:
            return None
        
        rows.append(new_row)
        return backtrack_generate(new_used, rows, row_idx + 1, max_val_bound)

    def local_search(rows, used_diffs):
        """
        Try to reduce the maximum value.
        """
        if not rows:
            return rows, used_diffs
            
        rows = [r[:] for r in rows]
        max_iter = 200
        for _ in range(max_iter):
            if time.time() - start_time > time_limit - 2:
                break
                
            # Find the row with the largest max
            max_val = max(max(r) for r in rows)
            target_idx = -1
            for i, r in enumerate(rows):
                if max(r) == max_val:
                    target_idx = i
                    break
            
            if target_idx == -1:
                break
                
            target_row = rows[target_idx]
            # Try to reduce the largest element (last element)
            last_elem = target_row[-1]
            
            # Find the second largest
            if len(target_row) < 2:
                break
            second_last = target_row[-2]
            
            # Try to replace last_elem with x where second_last < x < last_elem
            # We need to update used_diffs.
            # Diffs removed: last_elem - a for a in target_row[:-1]
            # Diffs added: x - a for a in target_row[:-1]
            
            old_diffs = {last_elem - a for a in target_row[:-1]}
            
            found_improvement = False
            # Try smaller x first
            for x in range(second_last + 1, last_elem):
                new_diffs = {x - a for a in target_row[:-1]}
                
                # Check if new_diffs conflict with (used_diffs - old_diffs)
                # i.e., new_diffs & (used_diffs - old_diffs) must be empty
                conflict = False
                for d in new_diffs:
                    if d in used_diffs and d not in old_diffs:
                        conflict = True
                        break
                
                if not conflict:
                    # Valid move
                    used_diffs -= old_diffs
                    used_diffs |= new_diffs
                    target_row[-1] = x
                    found_improvement = True
                    break
            
            if not found_improvement:
                # Try to swap last element with an element in another row?
                # Or just break to restart?
                # Let's try to swap last_elem with a smaller element from another row if possible.
                # This is complex. Let's just break and rely on restarts.
                break
                
        return rows, used_diffs

    best_rows = None
    best_score = float('inf')
    
    # Run multiple times
    while time.time() - start_time < time_limit:
        used_diffs = set()
        rows = []
        
        # Generate
        res = backtrack_generate(used_diffs, rows, 0, 0)
        if res is None:
            continue
        
        # Improve
        rows, used_diffs = local_search(rows, used_diffs)
        
        score = max(max(r) for r in rows)
        if score < best_score:
            best_score = score
            best_rows = [r[:] for r in rows]
            
            if best_score == 112: # Known optimal for 7,5
                break
            # If we find a very good score, we can tighten the search?
            # No, just keep going.
        
        # Randomize seed for next iteration?
        # random.seed() is not needed if we use random.choice internally.

    if best_rows is None:
        # Fallback: simple greedy
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        used_diffs.update(new_diffs)
                        break
                    x += 1
            rows.append(row)
        best_rows = rows

    return best_rows
