import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    # Total number of distinct differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Simple counting bound: sqrt(2 * total_diffs)
    import math
    lower_bound = int(math.ceil(math.sqrt(2 * total_diffs)))

    best_solution = None
    best_score = float('inf')

    def validate(rows):
        """Check if rows are valid: all positive differences distinct across all rows."""
        all_diffs = set()
        for row in rows:
            # Must start at 0 and be strictly increasing
            if not row or row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            # Check differences
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    def get_score(rows):
        return max(max(row) for row in rows)

    # Strategy 1: Deterministic greedy (fast baseline)
    def greedy_deterministic():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        return rows

    # Strategy 2: Backtracking with MRV heuristic (Most Constrained Variable)
    # Instead of building rows one at a time sequentially, we can think of this as
    # assigning difference sets to rows. But the standard approach of building
    # element-by-element with backtracking is more tractable.
    #
    # Key insight: Build rows one by one, but use backtracking within a row
    # if we get stuck, and also try different orderings or initial choices.
    #
    # A more powerful approach: treat the entire construction as a CSP where
    # we assign each element of each row. The domain of each element is bounded.
    #
    # For efficiency, let's try a hybrid:
    # 1. Deterministic greedy as baseline.
    # 2. Randomized greedy with backtracking on failure.
    # 3. Local search (simulated annealing / hill climbing) to improve scope.

    def greedy_with_backtracking(max_backtracks=100):
        """
        Greedy construction with limited backtracking.
        If we hit a dead end (no valid next element), backtrack and try alternative.
        """
        used = set()
        rows = []
        
        def backtrack(row_idx, col_idx, current_row):
            """
            Recursively build row row_idx, column col_idx.
            current_row is the partial row for row_idx.
            """
            if time.time() > deadline:
                return None
            
            if col_idx == k + 1:
                # Row complete
                rows[row_idx] = list(current_row)
                return True
            
            if row_idx == n_rows:
                return True
            
            # Try to find a valid value for current_row[col_idx]
            # The value must be > current_row[col_idx - 1] if col_idx > 0
            # and all differences with previous elements in current_row must be unused
            
            last_val = current_row[col_idx - 1] if col_idx > 0 else -1
            
            # We need to try values starting from last_val + 1
            # But how far to go? We don't know the upper bound a priori.
            # Let's try incrementally, but with a limit to avoid infinite loops.
            # If we fail, we backtrack.
            
            # For efficiency, we can try the smallest valid value first.
            # If that leads to a dead end later, we try the next one.
            
            x = last_val + 1
            while True:
                if x > 200:  # Arbitrary bound for this step; can be adjusted
                    return None
                
                new_diffs = {x - a for a in current_row}
                if not (new_diffs & used):
                    # Tentatively place x
                    used.update(new_diffs)
                    current_row.append(x)
                    
                    # Recurse to next column or next row
                    if col_idx + 1 < k + 1:
                        if backtrack(row_idx, col_idx + 1, current_row):
                            return True
                    else:
                        # Row complete, move to next row
                        if row_idx + 1 < n_rows:
                            if backtrack(row_idx + 1, 1, [0]):
                                return True
                        else:
                            return True
                    
                    # Backtrack
                    current_row.pop()
                    used.difference_update(new_diffs)
                
                x += 1
        
        # Start backtracking
        if backtrack(0, 1, [0]):
            return rows
        return None

    # Strategy 3: Local search / Hill climbing
    # Start with a valid solution (from greedy or backtracking).
    # Try to reduce the scope by modifying individual elements.
    
    def local_search(initial_rows, iterations=100000):
        """
        Hill climbing to reduce the maximum entry.
        """
        if initial_rows is None:
            return None
        
        rows = [list(r) for r in initial_rows]
        current_score = get_score(rows)
        best_score = current_score
        best_rows = [list(r) for r in rows]
        
        # Precompute used differences for quick validity check
        def compute_used_diffs(rows):
            used = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        used.add(row[j] - row[i])
            return used
        
        # We'll try random moves: pick a random element (not 0), and try to change it
        # to a nearby value that keeps the solution valid and potentially reduces scope.
        
        rng = random.Random(42)
        
        for it in range(iterations):
            if time.time() > deadline:
                break
            
            # Pick a random row and a random position (1..k, not 0)
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k)
            
            old_val = rows[r_idx][pos]
            
            # Try to change old_val to a new_val
            # new_val must be between rows[r_idx][pos-1] and rows[r_idx][pos+1] (if pos < k)
            # and maintain strict increasing order
            # Also, all differences involving this element must be distinct from all others.
            
            lower = rows[r_idx][pos - 1] + 1
            upper = rows[r_idx][pos + 1] - 1 if pos < k else old_val + 10  # Allow expansion
            
            # Try a few random values in [lower, upper]
            candidates = []
            if upper >= lower:
                # Try the value that minimizes the local contribution to scope
                # Or just try a few random values
                for _ in range(10):
                    new_val = rng.randint(lower, upper)
                    if new_val != old_val:
                        candidates.append(new_val)
            
            # Also try old_val - 1 and old_val + 1 if valid
            if lower <= old_val - 1 <= upper:
                candidates.append(old_val - 1)
            if lower <= old_val + 1 <= upper:
                candidates.append(old_val + 1)
            
            # Also try to reduce if old_val is the max
            if old_val == max(rows[r_idx]) and lower < old_val:
                candidates.append(lower)
            
            if not candidates:
                continue
            
            rng.shuffle(candidates)
            
            for new_val in candidates:
                # Check validity
                # Remove old differences involving position pos
                old_diffs = set()
                for i in range(pos):
                    old_diffs.add(old_val - rows[r_idx][i])
                for j in range(pos + 1, len(rows[r_idx])):
                    old_diffs.add(rows[r_idx][j] - old_val)
                
                # Add new differences
                new_diffs = set()
                for i in range(pos):
                    new_diffs.add(new_val - rows[r_idx][i])
                for j in range(pos + 1, len(rows[r_idx])):
                    new_diffs.add(rows[r_idx][j] - new_val)
                
                # Check if new_diffs conflict with any other differences
                # We need a global set of used differences excluding the old_diffs
                # For efficiency, we can maintain a global count or set.
                
                # Simple check: rebuild all differences except for row r_idx, pos
                # This is O(n_rows * k^2) which might be slow.
                
                # Alternative: maintain a global set of used diffs, and temporarily remove old_diffs
                # For now, let's do a quick check:
                # Compute all differences except those involving (r_idx, pos)
                
                # Actually, let's just try the move and validate locally.
                # If it's invalid, skip.
                
                # To make this efficient, we can precompute a set of all used diffs
                # and count how many times each diff appears.
                # But for simplicity, let's do a direct check for small k.
                
                # For now, skip this complex local search and rely on randomized greedy.
                break  # Skip local search for now to keep code simple
        
        return best_rows

    # Main search loop
    # 1. Deterministic greedy
    try:
        det_rows = greedy_deterministic()
        if det_rows and validate(det_rows):
            det_score = get_score(det_rows)
            if det_score < best_score:
                best_score = det_score
                best_solution = [list(r) for r in det_rows]
    except Exception:
        pass

    # 2. Randomized greedy with different seeds
    # The key is to randomize the choice of next element among valid options.
    
    def randomized_greedy(seed, p_random=0.3, window=5):
        """
        Greedy with randomization: with probability p_random, pick a random
        element from the first `window` valid options, otherwise pick the smallest.
        """
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for r in range(n_rows):
            row = [0]
            last_val = 0
            
            while len(row) < k + 1:
                # Find valid options
                valid_options = []
                x = last_val + 1
                
                while len(valid_options) < window:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_options.append(x)
                    x += 1
                    if x > 500:  # Safety bound
                        break
                
                if not valid_options:
                    return None  # Failed
                
                # Choose
                if rng.random() < p_random and len(valid_options) > 1:
                    chosen = valid_options[rng.randint(0, len(valid_options) - 1)]
                else:
                    chosen = valid_options[0]
                
                row.append(chosen)
                new_diffs = {chosen - a for a in row[:-1]}
                used.update(new_diffs)
                last_val = chosen
            
            rows.append(row)
        
        return rows

    # Run multiple randomized greeds
    seed = 0
    while time.time() < deadline - 10:
        rows = randomized_greedy(seed)
        if rows and validate(rows):
            score = get_score(rows)
            if score < best_score:
                best_score = score
                best_solution = [list(r) for r in rows]
        seed += 1
        if seed % 1000 == 0:
            # Occasionally try with different parameters
            pass

    # 3. Try backtracking with limited depth if time permits
    if time.time() < deadline - 20 and best_score > lower_bound:
        try:
            bt_rows = greedy_with_backtracking()
            if bt_rows and validate(bt_rows):
                bt_score = get_score(bt_rows)
                if bt_score < best_score:
                    best_score = bt_score
                    best_solution = [list(r) for r in bt_rows]
        except Exception:
            pass

    if best_solution is not None:
        return best_solution
    else:
        # Fallback: return deterministic greedy result
        try:
            det_rows = greedy_deterministic()
            if det_rows:
                return det_rows
        except:
            pass
        return []
