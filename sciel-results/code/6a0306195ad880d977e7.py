import random
import time
import math
from collections import Counter

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on max value: we need at least total_diffs distinct positive differences,
    # and the max difference in any row is at most max_val. But differences can overlap across rows.
    # Actually, all differences across all rows must be distinct, so we need at least total_diffs
    # distinct values in [1, max_val]. So max_val >= total_diffs.
    # But that's a very loose bound. The counting bound given is 112 for (4,7).
    # Let's use a tighter bound: each row with k+1 elements has k*(k+1)/2 differences.
    # The largest difference in a row is at most max_val.
    # A better lower bound: consider that differences in a single row range from 1 to max_val.
    # With n_rows rows, we need n_rows * k*(k+1)/2 distinct differences, each between 1 and max_val.
    # So max_val >= n_rows * k*(k+1)/2.
    # For (4,7): 4 * 7*8/2 = 4*28 = 112. That matches the counting bound!
    
    lower_bound = total_diffs

    # Greedy initial construction
    def greedy_initial():
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
        return rows, used

    rows, used_diffs = greedy_initial()
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    current_max = best_max

    def get_max(rows):
        return max(max(r) for r in rows)

    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    def rebuild_row_bitset(k, forbidden_diffs, rng, max_val_limit):
        """
        Build a row using bitset-based constraint propagation.
        We use a bitset to track which differences are forbidden.
        At each step, we find the smallest valid x such that all new differences
        (x - a for a in row) are not in the forbidden set.
        """
        # Convert forbidden_diffs to a bitset for O(1) lookup
        # We'll use a set for simplicity but with early termination
        row = [0]
        local_used = set()
        
        for step in range(k):
            last = row[-1]
            found = False
            
            # Try candidates in order from last+1 upward
            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                
                # Check if any new difference conflicts
                if new_diffs & local_used or new_diffs & forbidden_diffs:
                    continue
                
                # Check if we have enough room for remaining elements
                remaining = k - step - 1
                if remaining > 0:
                    min_gap_needed = remaining * (remaining + 1) // 2
                    if max_val_limit - x < min_gap_needed:
                        break  # No more valid candidates possible
                
                row.append(x)
                local_used.update(new_diffs)
                found = True
                break
            
            if not found:
                return None
        
        return row

    def rebuild_row_with_heuristic(k, forbidden_diffs, rng, max_val_limit):
        """
        Build a row with a heuristic that considers the impact of each choice
        on future flexibility. We score candidates by how many future difference
        slots they would block.
        """
        row = [0]
        local_used = set()
        
        for step in range(k):
            last = row[-1]
            remaining = k - step  # number of more elements to place (including this one)
            
            valid_candidates = []
            
            # Collect a limited set of candidates to evaluate
            for x in range(last + 1, min(max_val_limit + 1, last + 50)):
                new_diffs = {x - a for a in row}
                
                if new_diffs & local_used or new_diffs & forbidden_diffs:
                    continue
                
                # Check if we have enough room for remaining elements
                rem = k - step - 1
                if rem > 0:
                    min_gap_needed = rem * (rem + 1) // 2
                    if max_val_limit - x < min_gap_needed:
                        break
                
                # Score: prefer smaller x, and penalize candidates that create
                # dense clusters of differences (which may block future options)
                score = x * 0.5
                for d in new_diffs:
                    # Check neighborhood density
                    for delta in (-1, 1):
                        if (d + delta) in forbidden_diffs or (d + delta) in local_used:
                            score += 1.0
                
                valid_candidates.append((x, new_diffs, score))
                
                if len(valid_candidates) >= 10:
                    break
            
            if not valid_candidates:
                # Try a wider search
                for x in range(last + 1, max_val_limit + 1):
                    new_diffs = {x - a for a in row}
                    if new_diffs & local_used or new_diffs & forbidden_diffs:
                        continue
                    rem = k - step - 1
                    if rem > 0:
                        min_gap_needed = rem * (rem + 1) // 2
                        if max_val_limit - x < min_gap_needed:
                            break
                    valid_candidates.append((x, new_diffs, x * 0.5))
                    if len(valid_candidates) >= 5:
                        break
            
            if not valid_candidates:
                return None
            
            # Sort by score and pick from top candidates with some randomness
            valid_candidates.sort(key=lambda c: c[2])
            top_n = min(3, len(valid_candidates))
            idx = rng.randint(0, top_n - 1)
            x, new_diffs, _ = valid_candidates[idx]
            
            row.append(x)
            local_used.update(new_diffs)
        
        return row

    rng = random.Random(42)
    
    # Main search loop
    while time.time() < deadline:
        # Pick a row to rebuild
        i = rng.randint(0, n_rows - 1)
        
        old_row = rows[i]
        old_diffs = get_diffs(old_row)
        
        # Remove old row's differences from the global used set
        used_diffs.difference_update(old_diffs)
        
        # Try to rebuild with a tighter max
        new_row = None
        target_max = current_max - 1
        
        if target_max >= k:
            # Try bitset-based rebuild first
            new_row = rebuild_row_bitset(k, used_diffs, rng, target_max)
            if new_row is None:
                # Try heuristic rebuild
                new_row = rebuild_row_with_heuristic(k, used_diffs, rng, target_max)
        
        if new_row is None:
            # Fall back to current_max
            new_row = rebuild_row_bitset(k, used_diffs, rng, current_max)
            if new_row is None:
                new_row = rebuild_row_with_heuristic(k, used_diffs, rng, current_max)
        
        if new_row is None:
            # Restore old row
            used_diffs.update(old_diffs)
            continue
        
        new_diffs = get_diffs(new_row)
        
        # Verify no conflicts
        if new_diffs & used_diffs:
            used_diffs.update(old_diffs)
            continue
        
        # Accept the new row
        rows[i] = new_row
        new_global_max = get_max(rows)
        
        if new_global_max < current_max:
            used_diffs.update(new_diffs)
            current_max = new_global_max
            best_rows = [r[:] for r in rows]
        else:
            # Neutral move: accept with some probability to escape local minima
            if rng.random() < 0.2:
                used_diffs.update(new_diffs)
            else:
                rows[i] = old_row
                used_diffs.update(old_diffs)
        
        # Check if we've reached the lower bound
        if current_max == lower_bound:
            break
        
        # Periodically try to rebuild all rows with a tighter max
        if rng.random() < 0.1:
            # Try a full rebuild with target_max
            target = current_max - 1
            if target >= k:
                new_rows = []
                temp_used = set()
                success = True
                for j in range(n_rows):
                    nr = rebuild_row_bitset(k, temp_used, rng, target)
                    if nr is None:
                        success = False
                        break
                    new_rows.append(nr)
                    temp_used.update(get_diffs(nr))
                
                if success:
                    new_max = max(max(r) for r in new_rows)
                    if new_max < current_max:
                        rows = new_rows
                        used_diffs = temp_used
                        current_max = new_max
                        best_rows = [r[:] for r in rows]
    
    return best_rows

# EVOLVE-BLOCK-END
