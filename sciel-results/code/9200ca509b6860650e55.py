import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_diffs = n_rows * k * (k + 1) // 2
    
    def get_scope(rows):
        return max(max(r) for r in rows)

    def compute_used_set(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def greedy_construct(seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                # Search for next element
                # Try small increments first to keep it compact
                # But randomize to avoid getting stuck in bad local minima
                # We try a range of candidates
                
                # To be faster, we can check candidates in a random order 
                # but biased towards smaller values?
                # Or just try 1..100 in random order?
                
                # Let's try a mix: 
                # 1. Try last+1, last+2, ... last+20 in random order
                # 2. If fail, try wider range
                
                candidates = list(range(last + 1, last + 50))
                rng.shuffle(candidates)
                
                for x in candidates:
                    new_diffs = [x - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    row.append(x)
                    used.update(new_diffs)
                    found = True
                    break
                
                if not found:
                    # Expand search
                    candidates = list(range(last + 1, last + 200))
                    rng.shuffle(candidates)
                    for x in candidates:
                        new_diffs = [x - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                    if not found:
                        return None
            rows.append(row)
        return rows

    # 1. Get initial valid solution
    best_rows = None
    best_scope = float('inf')
    
    for i in range(50):
        if time.time() > deadline - 10:
            break
        rows = greedy_construct(i)
        if rows:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                
    if best_rows is None:
        # Fallback
        return [[0] * (k + 1) for _ in range(n_rows)] # Invalid but prevents crash

    current_rows = [list(r) for r in best_rows]
    current_scope = best_scope
    used_set = compute_used_set(current_rows)

    # 2. Compaction-based Local Search
    # Strategy: 
    # - Iterate through all elements (row by row, index by index).
    # - For each element, try to decrease its value to the smallest valid value.
    # - "Valid" means it doesn't conflict with existing diffs and maintains order.
    # - If a decrease is found, apply it and update used_set.
    # - Repeat until no changes (local optimum).
    # - Then perturb (randomly increase a random element) to escape.
    
    def try_compact(rows, used_set):
        """
        Attempts to reduce the scope of rows by moving elements to smaller values.
        Returns True if any change was made.
        """
        changed = False
        # We iterate in a specific order to ensure consistency.
        # Reverse order of rows and reverse order of indices might be better for compaction?
        # Let's try forward rows, forward indices.
        # Actually, to push elements left, we should process from the end?
        # If we move row[i][j] left, it might affect row[i][j+1].
        # If we process j from k down to 1, we can push j+1 left later?
        # No, j+1 is to the right. If we push j left, j+1 can also be pushed left.
        # So processing j from k down to 1 allows us to push the last element left, 
        # then the second to last, etc. This seems good.
        
        for r_idx in range(n_rows):
            row = rows[r_idx]
            # Process indices from last to first (excluding 0)
            for e_idx in range(len(row) - 1, 0, -1):
                old_val = row[e_idx]
                min_val = row[e_idx - 1] + 1
                # We want to find the smallest v >= min_val such that
                # 1. v < row[e_idx + 1] (if exists)
                # 2. Diffs are valid
                
                max_val = row[e_idx + 1] - 1 if e_idx < len(row) - 1 else old_val
                
                # We only care about reducing scope, so we only look at v < old_val
                # If old_val is the max of the row, reducing it reduces scope.
                # If old_val is not the max, reducing it might allow others to reduce.
                
                # Optimization: Only try to reduce if old_val > min_val
                if old_val <= min_val:
                    continue
                
                # Try values from min_val up to old_val - 1
                # We want the SMALLEST valid value to maximize compaction potential for subsequent moves
                found_valid = False
                for new_val in range(min_val, old_val):
                    # Check bounds
                    if e_idx < len(row) - 1 and new_val >= row[e_idx + 1]:
                        continue # This shouldn't happen if we break early, but just in case
                    
                    # Calculate removed and added diffs
                    removed_diffs = set()
                    for i in range(len(row)):
                        if i == e_idx:
                            continue
                        d = abs(row[i] - old_val)
                        removed_diffs.add(d)
                    
                    added_diffs = set()
                    # Temporarily set row[e_idx] to new_val for diff calculation?
                    # Or just calculate manually.
                    for i in range(len(row)):
                        if i == e_idx:
                            continue
                        d = abs(row[i] - new_val)
                        added_diffs.add(d)
                    
                    # Check for duplicates within the row's added diffs
                    # Actually, added_diffs is a set, so duplicates are merged.
                    # But we need to ensure that the NEW diffs don't collide with each other?
                    # No, if two pairs produce the same diff, it's invalid.
                    # Example: row = [0, 2, 4]. Change 2 to 1. Row [0, 1, 4]. Diffs: 1, 4, 3. Distinct.
                    # Example: row = [0, 3, 6]. Change 3 to 2. Row [0, 2, 6]. Diffs: 2, 6, 4. Distinct.
                    # Example: row = [0, 4, 8]. Change 4 to 2. Row [0, 2, 8]. Diffs: 2, 8, 6. Distinct.
                    # What if row = [0, 2, 4, 8]? Change 2 to 1. Row [0, 1, 4, 8].
                    # Diffs: 1, 4, 8, 3, 7, 4. Duplicate 4!
                    # So we MUST check for internal duplicates.
                    
                    # Efficient check for internal duplicates:
                    # The new row is row[:e_idx] + [new_val] + row[e_idx+1:]
                    # We already know the diffs NOT involving e_idx are valid (from previous state).
                    # We only need to check diffs involving new_val.
                    # But wait, if we change e_idx, do we affect diffs between other elements? No.
                    # So we only need to check if any added_diff appears twice in added_diffs?
                    # No, added_diffs is a set of values. If the same difference value appears from two different pairs, it's a conflict.
                    # So we need to count occurrences? Or just check if len(added_diffs) == number of pairs involving e_idx?
                    # Number of pairs involving e_idx is len(row) - 1.
                    # If len(added_diffs) < len(row) - 1, then there's a duplicate.
                    
                    if len(added_diffs) != len(row) - 1:
                        continue
                        
                    # Check against global used set
                    # new_used = used_set - removed_diffs
                    # if added_diffs & new_used: invalid
                    
                    # Optimization: Check intersection first
                    if added_diffs & (used_set - removed_diffs):
                        continue
                        
                    # Valid!
                    # Apply change
                    row[e_idx] = new_val
                    used_set = (used_set - removed_diffs) | added_diffs
                    changed = True
                    found_valid = True
                    break # We found the smallest valid value, so we stop searching for this element
                    
                # If no valid smaller value found, we leave it as is
                
        return changed, used_set

    # Main ILS Loop
    iterations = 0
    max_iterations = 2000 # Hard cap, but time is the real limit
    
    while time.time() < deadline - 2 and iterations < max_iterations:
        iterations += 1
        
        # 1. Compact
        changed, used_set = try_compact(current_rows, used_set)
        
        if changed:
            new_scope = get_scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
                # Restart perturbation counter?
        
        # 2. Perturb if we are stuck or to explore
        # We should perturb if we haven't improved in the last few iterations
        # Or just always perturb after a full compaction pass?
        # Let's perturb every time to ensure we explore.
        
        if time.time() > deadline - 5:
            break
            
        # Perturb: Pick a random element and increase it to a random valid value
        r_idx = random.randint(0, n_rows - 1)
        e_idx = random.randint(1, len(current_rows[r_idx]) - 1)
        old_val = current_rows[r_idx][e_idx]
        
        # Find a valid new value > old_val
        # Search range: old_val + 1 to old_val + 50
        found_perturb = False
        for _ in range(20): # Try a few times
            new_val = random.randint(old_val + 1, old_val + 50)
            
            # Check bounds
            if e_idx < len(current_rows[r_idx]) - 1 and new_val >= current_rows[r_idx][e_idx + 1]:
                continue
            
            removed_diffs = set()
            for i in range(len(current_rows[r_idx])):
                if i == e_idx:
                    continue
                d = abs(current_rows[r_idx][i] - old_val)
                removed_diffs.add(d)
            
            added_diffs = set()
            for i in range(len(current_rows[r_idx])):
                if i == e_idx:
                    continue
                d = abs(current_rows[r_idx][i] - new_val)
                added_diffs.add(d)
            
            if len(added_diffs) != len(current_rows[r_idx]) - 1:
                continue
                
            if added_diffs & (used_set - removed_diffs):
                continue
                
            # Valid perturbation
            current_rows[r_idx][e_idx] = new_val
            used_set = (used_set - removed_diffs) | added_diffs
            found_perturb = True
            break
            
        if not found_perturb:
            # If we can't perturb, we might be in a very tight spot.
            # Try a different element?
            # For now, just break or try next iteration.
            pass

    return best_rows

if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        print(f"Rows: {result}")
        # Validate
        used = set()
        valid = True
        for row in result:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        valid = False
                        print(f"Duplicate diff {d}")
                    used.add(d)
        print(f"Valid: {valid}")
    else:
        print("No valid construction found")
