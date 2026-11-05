"""
Strategy: Conflict-Driven Local Search with Heuristic Move Selection.

The previous approach used generic simulated annealing on row elements.
This approach:
1. Maintains a global set of differences and a map from difference to list of (row, i, j) tuples.
2. Identifies conflicts (differences with count > 1).
3. For a random conflict, picks one of the two (or more) occurrences.
4. Tries to move one of the endpoints of that difference to a new value that:
   a. Keeps the row sorted.
   b. Does not introduce new conflicts (or minimizes them).
   c. Ideally reduces the max entry.
5. Uses a "safe move" heuristic: precompute which values are "free" (not causing conflicts) 
   for each position, or just test a few candidates.
6. Iterates until time runs out.
"""

import random
import time
import math
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Total number of differences
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Helper to compute max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    # Helper to generate a random valid row within a bound
    def generate_random_row(max_val):
        if max_val < k:
            return list(range(k+1))
        vals = random.sample(range(1, max_val + 1), k)
        vals.sort()
        return [0] + vals
    
    # Initial solution: Greedy construction
    def pure_greedy():
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

    best_rows = pure_greedy()
    best_max = get_max_entry(best_rows)
    
    # State representation for fast conflict checking
    # We will maintain:
    # 1. rows: list of lists
    # 2. diff_count: dict {diff_value: count}
    # 3. diff_occurrences: dict {diff_value: list of (row_idx, i, j)}
    
    def build_state(rows):
        diff_count = defaultdict(int)
        diff_occurrences = defaultdict(list)
        for r_idx, row in enumerate(rows):
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    diff_count[d] += 1
                    diff_occurrences[d].append((r_idx, i, j))
        return diff_count, diff_occurrences

    current_rows = [row[:] for row in best_rows]
    diff_count, diff_occurrences = build_state(current_rows)
    
    # List of current conflicts
    def get_conflicts():
        conflicts = []
        for d, occs in diff_occurrences.items():
            if len(occs) > 1:
                conflicts.append((d, occs))
        return conflicts
    
    # Check if valid
    def is_valid(diff_count):
        return all(c == 1 for c in diff_count.values())
    
    # Update state after a move: rows[r_idx][pos] changes from old_val to new_val
    # We need to remove old diffs involving pos and add new ones.
    def update_state(r_idx, pos, old_val, new_val):
        row = current_rows[r_idx]
        
        # Remove old differences involving pos
        # Diffs are (i, j) where i < j.
        # pos can be i or j.
        # If pos is i, j > pos. If pos is j, i < pos.
        
        # Identify all pairs (i, j) involving pos
        pairs_to_remove = []
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                if i == pos or j == pos:
                    d = row[j] - row[i]
                    pairs_to_remove.append((d, i, j))
                    
        # Decrement counts and remove occurrences
        for d, i, j in pairs_to_remove:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
                if d in diff_occurrences:
                    # Remove (r_idx, i, j) from the list
                    # This is O(n) but n is small (total_diffs ~ 28*49=1372 max)
                    # Actually, for efficiency, we should just mark or rebuild if needed.
                    # Given the constraints, O(n) removal from list is acceptable for now.
                    occ_list = diff_occurrences[d]
                    for k in range(len(occ_list)):
                        if occ_list[k] == (r_idx, i, j):
                            occ_list.pop(k)
                            break
                    if not occ_list:
                        del diff_occurrences[d]
            else:
                # Just remove the occurrence from the list
                occ_list = diff_occurrences[d]
                for k in range(len(occ_list)):
                    if occ_list[k] == (r_idx, i, j):
                        occ_list.pop(k)
                        break

        # Apply the change
        row[pos] = new_val
        
        # Add new differences involving pos
        pairs_to_add = []
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                if i == pos or j == pos:
                    d = row[j] - row[i]
                    pairs_to_add.append((d, i, j))
                    
        for d, i, j in pairs_to_add:
            diff_count[d] += 1
            diff_occurrences[d].append((r_idx, i, j))
            
    # Main loop
    T = 100.0
    cooling = 0.9995
    last_improvement = start_time
    
    while time.time() < deadline:
        conflicts = get_conflicts()
        
        if not conflicts:
            # Valid solution
            curr_max = get_max_entry(current_rows)
            if curr_max < best_max:
                best_max = curr_max
                best_rows = [row[:] for row in current_rows]
                last_improvement = time.time()
                # Try to improve further by reducing max?
                # If valid, we can try to shrink the largest element.
                # Find the row with the max element
                max_row_idx = -1
                max_val = 0
                for r_idx, row in enumerate(current_rows):
                    if max(row) > max_val:
                        max_val = max(row)
                        max_row_idx = r_idx
                
                # Try to reduce max_val by moving the last element of max_row_idx down?
                # Or any element that contributes to the max.
                # Actually, if valid, any move that keeps it valid and reduces max is good.
                # Let's try to move the largest element in the largest row to a smaller value.
                # But we must maintain strict increase.
                
                # For now, just break or continue trying random moves?
                # If we are valid, we should try to minimize max.
                # Let's do a local search specifically for max reduction while maintaining validity.
                improved = False
                for _ in range(100): # Try a few times
                    # Pick a random row with max == best_max (or close)
                    # Actually, just pick the row with the global max
                    # Try to reduce its last element
                    # Or pick a random element and try to reduce it if it doesn't break validity
                    r_idx = max_row_idx
                    # Try to reduce the last element
                    pos = k
                    old_val = current_rows[r_idx][pos]
                    # Can we reduce it?
                    # It must be > current_rows[r_idx][pos-1]
                    min_new_val = current_rows[r_idx][pos-1] + 1
                    if old_val > min_new_val:
                        # Try a random value in [min_new_val, old_val - 1]
                        new_val = random.randint(min_new_val, old_val - 1)
                        # Check if this move keeps validity
                        # We need to simulate the move
                        # Save state
                        old_diff_count = dict(diff_count)
                        old_diff_occurrences = {k: list(v) for k, v in diff_occurrences.items()}
                        
                        update_state(r_idx, pos, old_val, new_val)
                        
                        if is_valid(diff_count):
                            improved = True
                            # Update best
                            curr_max = get_max_entry(current_rows)
                            if curr_max < best_max:
                                best_max = curr_max
                                best_rows = [row[:] for row in current_rows]
                                last_improvement = time.time()
                        else:
                            # Revert
                            update_state(r_idx, pos, new_val, old_val)
                    
                    if improved:
                        break
                    # Try other positions?
                    # For simplicity, just break if no improvement in last element
                    # Or try a random position in that row
                    pos = random.randint(1, k)
                    old_val = current_rows[r_idx][pos]
                    min_new_val = current_rows[r_idx][pos-1] + 1
                    max_new_val = current_rows[r_idx][pos+1] - 1 if pos < k else old_val
                    if old_val > min_new_val:
                        new_val = random.randint(min_new_val, min(old_val - 1, max_new_val))
                        if new_val == old_val:
                            continue
                        old_diff_count = dict(diff_count)
                        old_diff_occurrences = {k: list(v) for k, v in diff_occurrences.items()}
                        
                        update_state(r_idx, pos, old_val, new_val)
                        
                        if is_valid(diff_count):
                            improved = True
                            curr_max = get_max_entry(current_rows)
                            if curr_max < best_max:
                                best_max = curr_max
                                best_rows = [row[:] for row in current_rows]
                                last_improvement = time.time()
                        else:
                            update_state(r_idx, pos, new_val, old_val)
                    
                    if improved:
                        break

                if not improved:
                    # Stuck in a local minimum of valid solutions?
                    # Try to escape by allowing a temporary invalid move?
                    # Or just restart?
                    # For now, let's try a random move that might break validity but could lead to a better max later?
                    # No, let's just continue the main loop which will pick conflicts if we break validity.
                    # But we are currently valid.
                    # Let's try a random move that is likely to break validity but explores the space.
                    r_idx = random.randint(0, n_rows - 1)
                    pos = random.randint(1, k)
                    old_val = current_rows[r_idx][pos]
                    min_new_val = current_rows[r_idx][pos-1] + 1
                    max_new_val = best_max - 1 # Try to stay within current best
                    if pos < k:
                        max_new_val = min(max_new_val, current_rows[r_idx][pos+1] - 1)
                    
                    if min_new_val <= max_new_val:
                        new_val = random.randint(min_new_val, max_new_val)
                        if new_val != old_val:
                            old_diff_count = dict(diff_count)
                            old_diff_occurrences = {k: list(v) for k, v in diff_occurrences.items()}
                            update_state(r_idx, pos, old_val, new_val)
                            # Don't revert, let the main loop handle conflicts
                            # This effectively restarts the conflict resolution process
                            T = max(T, 50.0) # Reheat
                            continue
                    else:
                        break # No moves possible?
        
        if conflicts:
            # Pick a random conflict
            d, occs = random.choice(conflicts)
            # Pick one occurrence to fix
            r_idx, i, j = random.choice(occs)
            
            # We want to move one of the endpoints (i or j) to a new value.
            # Moving i affects diffs (i, j) and (i, j') for j' > i.
            # Moving j affects diffs (i, j) and (i', j) for i' < j.
            
            # Heuristic: Try to move the element that has more "freedom" or is larger?
            # Let's try moving j (the larger index) first, as it's often the "tail".
            # Or random.
            if random.random() < 0.5:
                pos = j
            else:
                pos = i
            
            row = current_rows[r_idx]
            old_val = row[pos]
            
            # Determine range for new_val
            min_new_val = row[pos-1] + 1 if pos > 0 else 1
            max_new_val = row[pos+1] - 1 if pos < k else best_max - 1
            
            if min_new_val > max_new_val:
                T *= cooling
                continue
                
            # Try a few candidate values
            # Prefer values that are "smaller" if we are trying to reduce max,
            # but first priority is resolving conflict.
            # Let's try random values in range.
            found_move = False
            for _ in range(10):
                new_val = random.randint(min_new_val, max_new_val)
                if new_val == old_val:
                    continue
                
                # Check if this move resolves the conflict without creating new ones
                # We need to simulate.
                # To avoid full state update/revert overhead, we can do a "dry run" check.
                # But state update is fast enough.
                
                old_diff_count = dict(diff_count)
                old_diff_occurrences = {k: list(v) for k, v in diff_occurrences.items()}
                
                update_state(r_idx, pos, old_val, new_val)
                
                # Check if the specific conflict d is resolved
                # And if no new conflicts were created (or fewer)
                # Actually, just check if total conflicts decreased or stayed same?
                # Let's check if d is still a conflict.
                if d in diff_occurrences and len(diff_occurrences[d]) > 1:
                    # Conflict still exists
                    # Check if we created new conflicts
                    new_conflict_count = sum(1 for occs in diff_occurrences.values() if len(occs) > 1)
                    old_conflict_count = sum(1 for occs in old_diff_occurrences.values() if len(occs) > 1)
                    
                    # Accept if conflicts decreased, or if same but max reduced?
                    # For now, accept if conflicts decreased.
                    if new_conflict_count < old_conflict_count:
                        found_move = True
                        break
                    else:
                        # Revert
                        update_state(r_idx, pos, new_val, old_val)
                        continue
                else:
                    # Conflict d resolved
                    # Check for new conflicts
                    new_conflict_count = sum(1 for occs in diff_occurrences.values() if len(occs) > 1)
                    old_conflict_count = sum(1 for occs in old_diff_occurrences.values() if len(occs) > 1)
                    
                    if new_conflict_count <= old_conflict_count:
                        found_move = True
                        break
                    else:
                        # Revert
                        update_state(r_idx, pos, new_val, old_val)
                        continue
            
            if not found_move:
                # Try another occurrence or another conflict?
                # For now, just cool down
                T *= cooling
                continue
        
        T *= cooling

    return best_rows

# EVOLVE-BLOCK-END
