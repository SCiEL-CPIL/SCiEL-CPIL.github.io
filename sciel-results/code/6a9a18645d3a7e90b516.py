"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline: evaluator kills at 150s; use 130s as our working limit.
    DEADLINE = 130.0
    start_time = time.time()
    
    # Total differences per row: k*(k+1)/2
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Greedy baseline (fast, deterministic)
    def greedy_build(n_rows, k, rng):
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
        return rows, max(max(r) for r in rows)
    
    # Fast greedy with randomization
    def greedy_randomized(n_rows, k, rng):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try a few candidates randomly before falling back to sequential
                found = False
                for attempt in range(20):
                    x += 1
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
                        found = True
                        break
                if not found:
                    # Fallback: sequential search
                    while True:
                        x += 1
                        new = {x - a for a in row}
                        if not new & used:
                            row.append(x)
                            used |= new
                            break
            rows.append(row)
        return rows, max(max(r) for r in rows)
    
    # Backtracking with forward checking for a single row given used diffs
    def backtrack_row(used, k, rng, max_val):
        """Find one row of k+1 elements starting at 0, using only diffs not in 'used'.
        Returns (row, new_used_diffs) or (None, None) if impossible within max_val."""
        row = [0]
        
        def dfs(pos, current_max):
            if time.time() - start_time > DEADLINE:
                return None
            if len(row) == k + 1:
                return list(row)
            
            # Determine candidate next values
            start = row[-1] + 1
            # Upper bound: we need to fit remaining (k+1 - len(row)) elements
            # Each must be at least 1 apart, so max possible is current_max + (k+1-len(row))
            # But also constrained by max_val
            end = min(max_val, current_max + (k + 1 - len(row)))
            
            if start > end:
                return None
            
            # Randomize order for variety
            candidates = list(range(start, end + 1))
            rng.shuffle(candidates)
            
            for x in candidates:
                # Check all differences from x to existing row elements
                new_diffs = set()
                conflict = False
                for a in row:
                    d = x - a
                    if d in used or d in new_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                row.append(x)
                result = dfs(pos + 1, x)
                if result is not None:
                    return result
                row.pop()
            
            return None
        
        result = dfs(0, 0)
        if result is not None:
            # Compute all diffs for this row
            all_diffs = set()
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    all_diffs.add(result[j] - result[i])
            return result, all_diffs
        return None, None
    
    # Local search: try to reduce the max entry by regenerating rows
    def local_search(rows, k, rng):
        """Try to improve by regenerating one row at a time with a tighter max bound."""
        current_max = max(max(r) for r in rows)
        improved = True
        while improved and time.time() - start_time < DEADLINE - 5:
            improved = False
            # Collect all used diffs
            all_used = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        all_used.add(row[j] - row[i])
            
            # Try each row index
            for idx in range(n_rows):
                if time.time() - start_time > DEADLINE - 5:
                    break
                # Remove this row's diffs
                row = rows[idx]
                row_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        row_diffs.add(row[j] - row[i])
                
                remaining_used = all_used - row_diffs
                
                # Try to find a new row with a smaller max
                # Binary search on max_val
                lo = k  # minimum possible max for k+1 elements starting at 0
                hi = max(current_max - 1, k)
                
                best_row = None
                # Try from hi down to lo, but limit attempts
                attempts = 0
                for max_val in range(hi, lo - 1, -1):
                    if time.time() - start_time > DEADLINE - 5:
                        break
                    attempts += 1
                    if attempts > 50:
                        break
                    
                    new_row, new_diffs = backtrack_row(remaining_used, k, rng, max_val)
                    if new_row is not None:
                        best_row = new_row
                        break
                
                if best_row is not None:
                    rows[idx] = best_row
                    current_max = max(max(r) for r in rows)
                    improved = True
        
        return rows, current_max
    
    # Main logic
    # Start with greedy
    rng = random.Random(42)
    best_rows, best_max = greedy_build(n_rows, k, rng)
    
    # Try randomized greedy multiple times
    for trial in range(100):
        if time.time() - start_time > DEADLINE - 10:
            break
        rows, mx = greedy_randomized(n_rows, k, rng)
        if mx < best_max:
            best_rows, best_max = rows, mx
    
    # Local search to improve
    rows, mx = local_search(list(best_rows), k, rng)
    if mx < best_max:
        best_rows, best_max = rows, mx
    
    # Additional randomized restarts with backtracking
    for trial in range(200):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Try to build all rows with backtracking, aiming for a smaller max
        # Start with a target max and try to achieve it
        target_max = best_max - 1
        if target_max < k:
            break
        
        # Try to build rows sequentially with backtracking
        used = set()
        new_rows = []
        success = True
        
        for i in range(n_rows):
            if time.time() - start_time > DEADLINE - 10:
                success = False
                break
            
            row, diffs = backtrack_row(used, k, rng, target_max)
            if row is None:
                # Try with slightly larger max for this row
                for extra in range(1, 5):
                    if time.time() - start_time > DEADLINE - 10:
                        break
                    row, diffs = backtrack_row(used, k, rng, target_max + extra)
                    if row is not None:
                        break
                if row is None:
                    success = False
                    break
            
            new_rows.append(row)
            used |= diffs
        
        if success:
            new_max = max(max(r) for r in new_rows)
            if new_max < best_max:
                best_rows, best_max = new_rows, new_max
    
    return best_rows
# EVOLVE-BLOCK-END
