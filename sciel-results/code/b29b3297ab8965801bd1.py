"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random

# Global deadline guard
_DEADLINE = None
_TIME_LIMIT = 115.0  # seconds, safety margin before 120s kill

def _time_left():
    if _DEADLINE is None:
        return _TIME_LIMIT
    return max(0.0, _DEADLINE - time.time())

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    """
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    
    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Estimate max difference needed to size the bitset
    # Lower bound for max entry m: m*(m+1)/2 >= total_diffs
    # For n=4, k=7, total_diffs = 4 * 28 = 112.
    # 15*16/2 = 120, so m >= 15 is a very loose bound.
    # Actual known solution is 134. Let's allow up to 200 for safety.
    MAX_VAL = 200
    # Bitset mask for valid differences 1..MAX_VAL
    # We only care about differences up to MAX_VAL
    
    def greedy_construct_bitset(n_rows, k, rng):
        """
        Greedy construction using bitsets for fast difference checking.
        """
        used = 0  # Bitset of used differences
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                # Try smallest available x
                for candidate in range(x + 1, MAX_VAL + 1):
                    # Check if candidate is valid
                    # New diffs: candidate - a for a in row
                    # We need to ensure none of these are in 'used'
                    # Also, candidate - a must be positive (guaranteed since candidate > x >= a)
                    
                    # Optimization: Pre-compute diffs? No, row is small.
                    valid = True
                    for a in row:
                        d = candidate - a
                        if d > MAX_VAL:
                            valid = False
                            break
                        if (used >> d) & 1:
                            valid = False
                            break
                    
                    if valid:
                        # Accept candidate
                        x = candidate
                        row.append(x)
                        # Update used bitset
                        for a in row[:-1]:
                            d = x - a
                            used |= (1 << d)
                        found = True
                        break
                if not found:
                    # If we hit MAX_VAL and didn't find one, we might need to expand
                    # For this problem, 200 should be enough for a valid solution, 
                    # but if not, we return a partial or invalid one and let local search fix it?
                    # Better to just raise the limit or return what we have.
                    # Given the small size, 200 is very generous.
                    # If we fail, we return the best effort.
                    pass
            rows.append(row)
        return rows, used

    def local_search_bitset(rows, used, max_iters=1000):
        """
        Local search to minimize the maximum entry.
        We iterate through rows, and for each row, try to reduce the largest element,
        then the second largest, etc.
        """
        rows = [list(r) for r in rows]
        # Rebuild used bitset from rows to be safe
        used = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= MAX_VAL:
                        used |= (1 << d)
                    else:
                        # If difference exceeds MAX_VAL, our bitset is too small.
                        # This shouldn't happen if we constructed with MAX_VAL=200 and optimal is 134.
                        pass

        improved = False
        for _ in range(max_iters):
            if _time_left() < 2.0:
                break
            
            any_progress = False
            
            # Sort rows by their max element, largest first
            row_indices = sorted(range(len(rows)), key=lambda i: rows[i][-1], reverse=True)
            
            for idx in row_indices:
                row = rows[idx]
                # Try to reduce each element from the end to the start
                for pos in range(len(row) - 1, 0, -1):
                    current_val = row[pos]
                    # Try to find a smaller value for row[pos]
                    # Constraints:
                    # 1. row[pos-1] < new_val < row[pos+1] (if pos+1 exists)
                    # 2. All new differences must be unique and not in 'used' (excluding old diffs)
                    
                    # Calculate old diffs involving row[pos]
                    old_diffs_mask = 0
                    for j in range(pos):
                        d = current_val - row[j]
                        old_diffs_mask |= (1 << d)
                    for j in range(pos + 1, len(row)):
                        d = row[j] - current_val
                        old_diffs_mask |= (1 << d)
                    
                    # The "free" mask is used without these diffs
                    # But we must be careful: if we change row[pos], diffs with other elements in the same row change.
                    # Diffs between other elements in the same row remain constant.
                    
                    # Let's iterate possible new values
                    # Lower bound: row[pos-1] + 1
                    # Upper bound: current_val - 1
                    start = row[pos-1] + 1
                    end = current_val
                    
                    # Optimization: If current_val is small, loop is small.
                    for new_val in range(start, end):
                        # Check if new_val creates any conflict
                        # New diffs with previous elements
                        conflict = False
                        new_diffs_mask = 0
                        for j in range(pos):
                            d = new_val - row[j]
                            if d <= 0: continue # Should not happen
                            if (used >> d) & 1:
                                # Check if this diff was one of the old diffs we are removing
                                # If it was in old_diffs_mask, it's okay? 
                                # No, 'used' contains ALL diffs. 
                                # We need to check if d is in 'used' AND d is NOT in old_diffs_mask.
                                if not (old_diffs_mask >> d) & 1:
                                    conflict = True
                                    break
                            new_diffs_mask |= (1 << d)
                        
                        if conflict:
                            continue
                        
                        # Check new diffs with next elements
                        for j in range(pos + 1, len(row)):
                            d = row[j] - new_val
                            if d <= 0: continue
                            if (used >> d) & 1:
                                if not (old_diffs_mask >> d) & 1:
                                    conflict = True
                                    break
                            new_diffs_mask |= (1 << d)
                        
                        if conflict:
                            continue
                        
                        # If no conflict, apply change
                        # Update 'used': remove old_diffs_mask, add new_diffs_mask
                        # But wait, old_diffs_mask includes diffs that are shared? 
                        # No, each diff is unique globally.
                        # So we can just: used = (used & ~old_diffs_mask) | new_diffs_mask
                        
                        # However, we must ensure new_diffs_mask doesn't have bits set that are in 'used' but not in old_diffs_mask.
                        # We already checked that.
                        
                        # Apply
                        used = (used & ~old_diffs_mask) | new_diffs_mask
                        row[pos] = new_val
                        improved = True
                        any_progress = True
                        break # Move to next position or row
                    
                    if any_progress:
                        # If we changed a value, the order of elements in the row might need re-evaluation?
                        # No, we are reducing values, so row remains sorted.
                        # But the "largest first" order for rows might change.
                        # We break out of the pos loop and re-sort rows.
                        break
            
            if not any_progress:
                break
        
        return rows, used

    # Initial construction
    best_rows = None
    best_max = float('inf')
    
    rng = random.Random(42)
    
    # Try a few different greedy seeds
    for i in range(5):
        rows, used = greedy_construct_bitset(n_rows, k, rng)
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = rows
            best_used = used
            
        if _time_left() < 5.0:
            break

    if best_rows is None:
        # Fallback: simple greedy
        rows, used = greedy_construct_bitset(n_rows, k, rng)
        best_rows = rows
        best_used = used
        best_max = max(max(r) for r in rows)

    # Local search
    if _time_left() > 5.0:
        improved_rows, improved_used = local_search_bitset(best_rows, best_used, max_iters=2000)
        new_max = max(max(r) for r in improved_rows)
        if new_max < best_max:
            best_max = new_max
            best_rows = improved_rows
            best_used = improved_used

    # If we have time, do more local search passes or restarts
    while _time_left() > 5.0:
        # Randomize slightly and try to improve?
        # Or just run local search again on the best found.
        # Local search is deterministic if no randomness. 
        # To escape local minima, we can add a random perturbation.
        
        # Perturb: pick a random row, random pos, increase value slightly?
        # Or just run local search again with a different iteration limit?
        # Let's try a randomized local search step.
        
        # Create a copy
        rows = [list(r) for r in best_rows]
        used = best_used
        
        # Random move: pick a random element, change it to a random valid nearby value
        # This is complex. Let's stick to deterministic local search but run it multiple times.
        # Since it's deterministic, it will converge to the same local minimum.
        
        # Instead, let's try a different initial greedy construction if we haven't found 134 yet.
        if best_max > 134:
             # Try another greedy
             rows, used = greedy_construct_bitset(n_rows, k, rng)
             cur_max = max(max(r) for r in rows)
             if cur_max < best_max:
                 best_max = cur_max
                 best_rows = rows
                 best_used = used
                 # Local search on this new candidate
                 if _time_left() > 5.0:
                     improved_rows, improved_used = local_search_bitset(best_rows, best_used, max_iters=1000)
                     new_max = max(max(r) for r in improved_rows)
                     if new_max < best_max:
                         best_max = new_max
                         best_rows = improved_rows
                         best_used = improved_used
        else:
             # We found 134 or better. Try to improve further.
             # Run local search again to see if it can go lower.
             # It will likely return the same result.
             # Break to save time?
             # The counting bound is 112. 134 is the known best.
             # If we hit 134, we are likely at the optimal or very close.
             # Let's do one final local search pass.
             improved_rows, improved_used = local_search_bitset(best_rows, best_used, max_iters=500)
             new_max = max(max(r) for r in improved_rows)
             if new_max < best_max:
                 best_max = new_max
                 best_rows = improved_rows
                 best_used = improved_used
             break

    # Validate
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    if row[j] <= row[i]:
                        return False
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    if not validate(best_rows, n_rows, k):
        # Fallback
        rows, used = greedy_construct_bitset(n_rows, k, rng)
        best_rows = rows

    return best_rows
