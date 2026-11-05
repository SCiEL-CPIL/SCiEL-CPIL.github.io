import random
import time

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit

    def time_up():
        return time.time() > deadline

    # Helper to compute all positive differences in a row
    def get_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def get_max(rows):
        return max(max(r) for r in rows)

    # 1. Greedy Initial Solution
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
    best_max = get_max(best_rows)
    current_max = best_max

    # Lower bound: total differences needed is n_rows * k*(k+1)//2
    # The max entry must be at least the (total_diffs)-th smallest possible difference?
    # Actually, the differences are distinct integers from 1..M.
    # So M >= n_rows * k*(k+1)//2.
    # But also each row has k+1 elements, so max entry >= k.
    lb = max(k, n_rows * k * (k + 1) // 2)
    
    # If we hit the lower bound, we can stop.
    if current_max == lb:
        return best_rows

    # 2. Local Search with Rebuilds
    rng = random.Random(42)
    
    def rebuild_row(k, forbidden_diffs, rng, max_val_limit, bias=0.0):
        """
        Builds a row of size k+1 starting at 0, strictly increasing,
        avoiding forbidden_diffs, with max entry <= max_val_limit.
        bias: probability to pick a random valid candidate from the first 10, else smallest.
        """
        row = [0]
        local_used = set()
        
        for _ in range(k):
            last = row[-1]
            valid_candidates = []
            
            # Search for valid candidates
            # Optimization: break early if we have enough candidates to choose from
            # and we are biased towards small ones.
            # But we must ensure we don't miss a "small" valid one if we break too early?
            # If we iterate x from last+1 upwards, the first valid ones are the smallest.
            # So collecting the first N valid ones gives us the N smallest valid x.
            
            # Let's collect up to 10 smallest valid x.
            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    valid_candidates.append((x, new_diffs))
                    if len(valid_candidates) >= 10:
                        break
            
            if not valid_candidates:
                return None
            
            # Selection strategy
            if bias > 0 and rng.random() < bias:
                # Pick randomly from the smallest 10
                idx = rng.randint(0, len(valid_candidates) - 1)
                x, new_diffs = valid_candidates[idx]
            else:
                # Pick the smallest (greedy)
                x, new_diffs = valid_candidates[0]
            
            row.append(x)
            local_used.update(new_diffs)
            
            if x > max_val_limit:
                return None
                
        return row

    # Main Loop
    no_improve_count = 0
    max_no_improve = 1000  # Reset strategy if stuck
    
    while not time_up():
        # Pick a row to rebuild
        i = rng.randint(0, n_rows - 1)
        
        old_row = rows[i]
        old_diffs = get_diffs(old_row)
        
        # Remove from used_diffs
        used_diffs.difference_update(old_diffs)
        
        # Determine target max for the new row
        # We want to reduce the global max.
        # If the row we are rebuilding is NOT the one with the global max, 
        # reducing its max won't reduce the global max unless it was tied?
        # Actually, if we reduce any row, the global max can only stay same or decrease.
        # To decrease global max, we must reduce a row that currently has max == current_max.
        # However, reducing a "small" row might free up differences that allow the "big" row to shrink later?
        # No, differences are forbidden. Freeing up differences makes it EASIER to build other rows.
        # So it's good to shuffle.
        
        # Strategy:
        # 1. Try to build a row with max < current_max.
        # 2. If that fails, try to build a row with max <= current_max (to shuffle).
        
        new_row = None
        
        # Attempt 1: Improve (strictly smaller max for this row)
        # We don't know if this row's max is the global max.
        # If old_row[-1] < current_max, we can try to keep it small.
        # If old_row[-1] == current_max, we MUST make it smaller to improve global max.
        
        target_max = current_max - 1
        if target_max > k:
            # Use a higher bias for exploration when trying to improve?
            # Or just greedy? Greedy is fast.
            new_row = rebuild_row(k, used_diffs, rng, target_max, bias=0.5)
            
        # Attempt 2: Maintain/Reshuffle (if improve failed)
        if new_row is None:
            # Try to build a valid row with max <= current_max
            # Use higher bias to explore different structures
            new_row = rebuild_row(k, used_diffs, rng, current_max, bias=0.8)
        
        if new_row is None:
            # Failed to build any valid row within current_max
            # Restore old row
            used_diffs.update(old_diffs)
            no_improve_count += 1
            continue
        
        new_diffs = get_diffs(new_row)
        
        # Verify validity (should be true by construction, but safe check)
        if new_diffs & used_diffs:
            used_diffs.update(old_diffs)
            no_improve_count += 1
            continue
            
        # Calculate new global max
        rows[i] = new_row
        new_global_max = get_max(rows)
        
        if new_global_max < current_max:
            # Improvement!
            used_diffs.update(new_diffs)
            current_max = new_global_max
            best_rows = [r[:] for r in rows]
            no_improve_count = 0
            
            if current_max == lb:
                break
        else:
            # No improvement in global max
            # Accept with probability to allow shuffling
            if rng.random() < 0.2:
                used_diffs.update(new_diffs)
                # current_max stays same
                no_improve_count += 1
            else:
                # Reject: restore old row
                rows[i] = old_row
                used_diffs.update(old_diffs)
                no_improve_count += 1
                
        # If stuck for a long time, try a different strategy?
        # For now, just keep going. The randomization in rebuild should eventually find a path.
        
    # Final Polish Pass
    # Try to reduce the max further with a tighter search
    if not time_up():
        polish_deadline = deadline - 5.0  # 5 seconds safety margin
        
        # Sort rows by max value, largest first
        row_indices = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)
        
        for i in row_indices:
            if time.time() > polish_deadline:
                break
                
            old_row = rows[i]
            old_diffs = get_diffs(old_row)
            
            # Try to reduce this row's max by 1, 2, 3...
            for reduction in range(1, 10):
                target_max = old_row[-1] - reduction
                if target_max <= k:
                    break
                    
                # Try to rebuild with this target
                # Use lower bias for more deterministic improvement
                new_row = rebuild_row(k, used_diffs.difference(old_diffs), rng, target_max, bias=0.1)
                
                if new_row is not None:
                    new_diffs = get_diffs(new_row)
                    if not (new_diffs & (used_diffs - old_diffs)):
                        # Success!
                        used_diffs.difference_update(old_diffs)
                        used_diffs.update(new_diffs)
                        rows[i] = new_row
                        current_max = get_max(rows)
                        best_rows = [r[:] for r in rows]
                        break
                        
    return best_rows

# EVOLVE-BLOCK-END
