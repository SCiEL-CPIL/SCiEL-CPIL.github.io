import time
import random
import sys

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set with n_rows rows of k+1 integers.
    Uses Simulated Annealing to minimize the largest entry.
    """
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s kill
    
    # Total differences per row is k*(k+1)/2
    # Total differences overall is n_rows * k*(k+1)/2
    
    def get_greedy_row(used_diffs, current_max_hint=0):
        """
        Constructs a single row greedily, trying to keep entries small.
        Returns the row and the set of differences used.
        """
        row = [0]
        diffs = set()
        # We want to find k more elements.
        # To keep max small, we should try to add elements close to 0 first? 
        # No, greedy ruler construction usually adds the smallest x such that x-a is not in used_diffs.
        # However, if we just do that, we might create a very long row.
        # A better heuristic for DTS is to try to pack differences densely.
        
        x = 1
        while len(row) < k + 1:
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                diffs.update(new_diffs)
                used_diffs.update(new_diffs)
                x += 1
            else:
                x += 1
                # Optimization: If x gets too large, break? 
                # But we must complete the row.
        return row, diffs

    def build_initial_solution():
        """
        Builds a valid solution using the simple greedy method from the seed.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row, _ = get_greedy_row(used_diffs)
            rows.append(row)
        return rows

    def get_max_val(rows):
        return max(max(row) for row in rows)

    def is_valid(rows):
        """
        Checks if the set of rows is valid (all differences distinct).
        """
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def local_improve(rows, used_diffs):
        """
        Attempts to reduce the maximum entry of each row by trying to replace
        the largest element with a smaller one that doesn't conflict.
        """
        improved = True
        while improved:
            improved = False
            for i in range(n_rows):
                row = rows[i]
                # Try to shrink the last element (largest)
                # Remove last element from used_diffs
                last_val = row[-1]
                row_diffs = {last_val - a for a in row[:-1]}
                used_diffs -= row_diffs
                
                # Try to find a smaller value for the last position
                # The new value must be > row[-2]
                # And its differences to row[:-1] must not be in used_diffs
                # Also, the new differences must not be in row_diffs of other rows (handled by used_diffs)
                
                # We want to minimize the new value.
                # Start from row[-2] + 1 up to last_val - 1
                found = False
                for new_val in range(row[-2] + 1, last_val):
                    new_diffs = {new_val - a for a in row[:-1]}
                    if not new_diffs & used_diffs:
                        # Check if new_diffs are distinct from each other? 
                        # Yes, because they are differences to distinct elements in row[:-1]
                        # and new_val is unique.
                        row[-1] = new_val
                        used_diffs |= new_diffs
                        improved = True
                        found = True
                        break
                if not found:
                    # Re-add original differences
                    used_diffs |= row_diffs
                # Note: If we improved, we continue the outer loop to check other rows
                # If we didn't improve, we move to next row

    def simulated_annealing(rows, used_diffs):
        """
        Main optimization loop.
        """
        best_rows = [r[:] for r in rows]
        best_score = get_max_val(rows)
        current_score = best_score
        
        # Temperature schedule
        T = 10.0
        cooling_rate = 0.995
        min_T = 0.1
        
        while time.time() - start_time < timeout:
            if T < min_T:
                break
            
            # Pick a random row to modify
            idx = random.randint(0, n_rows - 1)
            
            # Save state
            old_row = rows[idx][:]
            old_diffs = set()
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    old_diffs.add(old_row[j] - old_row[i])
            
            # Remove old row's differences
            used_diffs -= old_diffs
            
            # Rebuild this row greedily, but with some randomness or different strategy
            # Strategy: Greedy but allow skipping small x if it leads to large jumps later?
            # Simplest: Just re-greedy. If we get the same row, we wasted time.
            # Better: Randomized Greedy.
            new_row, new_diffs = get_randomized_greedy_row(used_diffs, k)
            
            if new_row is None:
                # Failed to build, revert
                used_diffs |= old_diffs
                continue
            
            rows[idx] = new_row
            used_diffs |= new_diffs
            
            # Local improvement on the new row and neighbors
            local_improve(rows, used_diffs)
            
            new_score = get_max_val(rows)
            
            # Acceptance criterion
            if new_score < current_score or random.random() < pow(2.71828, (current_score - new_score) / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_rows = [r[:] for r in rows]
            else:
                # Reject: Revert
                rows[idx] = old_row
                used_diffs -= new_diffs
                used_diffs |= old_diffs
                # Re-run local improve on reverted state to ensure consistency? 
                # Actually, local_improve is monotonic (only reduces max), so reverting 
                # to a previous state that was already locally improved is fine.
                # But wait, if we rejected, we are back to the state BEFORE the move.
                # That state was already locally improved in the previous iteration (or initially).
                # So no need to re-improve.
            
            T *= cooling_rate
            
            # Periodic global local improvement
            if random.random() < 0.05:
                local_improve(rows, used_diffs)
                current_score = get_max_val(rows)
                if current_score < best_score:
                    best_score = current_score
                    best_rows = [r[:] for r in rows]

        return best_rows

    def get_randomized_greedy_row(used_diffs, k):
        """
        Builds a row similar to greedy but with a bias towards smaller values
        or random jumps to escape local minima.
        """
        row = [0]
        diffs = set()
        x = 1
        max_x_limit = 200  # Safety limit for single row construction
        
        while len(row) < k + 1:
            # Try a few candidates around x
            # Standard greedy picks the first valid x.
            # Randomized: Pick a valid x from a small window [x, x + window_size]
            # or just pick the smallest valid x but occasionally jump?
            
            # Let's stick to smallest valid x for stability, 
            # but if we are stuck, we might need a different heuristic.
            # For now, standard greedy is a good baseline move.
            # To make it a "move", we can randomly permute the order of checking 
            # or introduce a small bias.
            
            # Actually, re-greedy-ing the same row with the same used_diffs 
            # will produce the exact same row. 
            # So we need to perturb the used_diffs or the row construction.
            
            # Alternative Move: Swap differences between two rows.
            # This is complex.
            
            # Let's try a different move: Shift the entire row.
            # If row is [0, a, b, ...], shifting by s gives [s, s+a, ...]
            # But the row MUST start at 0. So we can't shift the whole row.
            
            # Move: Replace one element.
            # Pick a random index j in the row (not 0, not last).
            # Remove row[j].
            # Try to insert a new value v > row[j-1] such that diffs are valid.
            
            # Let's implement "Replace Element" move instead of full rebuild.
            return None # Placeholder, we will implement replace_element_move in the main loop logic

    # Refined Plan:
    # 1. Initial Greedy
    # 2. Local Improve (Shrink max)
    # 3. SA Loop with "Replace Element" and "Rebuild Row" moves.

    # Re-implementing the core logic inside construct for clarity and speed

    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row, _ = get_greedy_row(used_diffs)
        rows.append(row)
    
    # Initial local improvement
    local_improve(rows, used_diffs)
    
    best_rows = [r[:] for r in rows]
    best_score = get_max_val(rows)
    current_score = best_score
    
    T = 5.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while time.time() - start_time < timeout:
        if T < min_T:
            # Reheat
            T = 5.0
            if get_max_val(best_rows) < 100: # If we are already good, stop?
                # The target is 112. If we are below, we are great.
                pass
        
        move_type = random.choice(['replace', 'rebuild'])
        
        idx = random.randint(0, n_rows - 1)
        
        if move_type == 'replace':
            # Pick a random position in the row to replace
            # Cannot replace index 0 (must be 0)
            # Replace index j (1 <= j <= k)
            j = random.randint(1, k)
            
            old_val = rows[idx][j]
            
            # Remove differences involving old_val
            row = rows[idx]
            to_remove = set()
            for i in range(len(row)):
                if i != j:
                    d = abs(row[j] - row[i])
                    to_remove.add(d)
            
            used_diffs -= to_remove
            
            # Try to find a new value for index j
            # Constraints:
            # 1. row[j-1] < new_val < row[j+1] (if j < k)
            # 2. Differences new_val - row[i] (i < j) and row[i] - new_val (i > j) must not be in used_diffs
            # 3. Differences must be distinct within the row (automatically if new_val is distinct from others)
            
            low = row[j-1] + 1
            high = row[j+1] - 1 if j < k else 1000 # Upper bound for search
            
            found = False
            # Try values close to old_val first? Or random?
            # Let's try a random range around old_val
            search_range = 20
            candidates = list(range(max(low, old_val - search_range), min(high, old_val + search_range) + 1))
            random.shuffle(candidates)
            
            for new_val in candidates:
                new_diffs = set()
                valid = True
                for i in range(len(row)):
                    if i == j:
                        continue
                    d = abs(new_val - row[i])
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    row[j] = new_val
                    used_diffs |= new_diffs
                    found = True
                    break
            
            if not found:
                # Revert
                row[j] = old_val
                used_diffs |= to_remove
                continue

        else: # rebuild
            # Remove entire row
            row = rows[idx]
            to_remove = set()
            for i in range(len(row)):
                for l in range(i+1, len(row)):
                    to_remove.add(row[l] - row[i])
            
            used_diffs -= to_remove
            
            # Rebuild greedily
            new_row, new_diffs = get_greedy_row(used_diffs)
            
            if new_row is None:
                used_diffs |= to_remove
                continue
                
            rows[idx] = new_row
            used_diffs |= new_diffs

        # Local improvement after every move
        local_improve(rows, used_diffs)
        
        new_score = get_max_val(rows)
        
        if new_score < current_score or random.random() < pow(2.71828, (current_score - new_score) / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_rows = [r[:] for r in rows]
        else:
            # Revert is tricky with 'replace' if we already modified row.
            # We need to store the full state before the move.
            # Optimization: Only accept if better, or with probability.
            # If rejected, we must restore the exact previous state.
            # This requires saving the row and used_diffs changes.
            # For 'replace', we saved old_val and to_remove.
            # For 'rebuild', we saved to_remove.
            # However, local_improve might have changed OTHER rows!
            # So simple revert is not enough if local_improve changed other rows.
            
            # To handle this correctly, we should snapshot the entire rows and used_diffs.
            # This is expensive but safe.
            # Given n_rows=7, k=6, rows are small. used_diffs is a set of ints.
            # Copying is fast.
            
            pass # We need to implement proper snapshotting below

    # The logic above has a flaw: local_improve modifies other rows, so rejection is hard.
    # Let's restart with a cleaner SA implementation using full snapshots.
    
    # --- Clean SA Implementation ---
    
    def snapshot(rows, used_diffs):
        return [r[:] for r in rows], set(used_diffs)
    
    def restore(rows, used_diffs, snap_rows, snap_diffs):
        for i in range(n_rows):
            rows[i] = snap_rows[i][:]
        used_diffs.clear()
        used_diffs.update(snap_diffs)

    # Re-initialize
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row, _ = get_greedy_row(used_diffs)
        rows.append(row)
    local_improve(rows, used_diffs)
    
    best_rows = [r[:] for r in rows]
    best_score = get_max_val(rows)
    current_score = best_score
    
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.1
    
    while time.time() - start_time < timeout:
        if T < min_T:
            break
            
        snap_rows, snap_diffs = snapshot(rows, used_diffs)
        
        # Pick a move
        move_type = random.choice(['replace', 'rebuild', 'swap_diffs'])
        
        if move_type == 'replace':
            idx = random.randint(0, n_rows - 1)
            j = random.randint(1, k)
            row = rows[idx]
            old_val = row[j]
            
            to_remove = set()
            for i in range(len(row)):
                if i != j:
                    to_remove.add(abs(row[j] - row[i]))
            
            used_diffs -= to_remove
            
            low = row[j-1] + 1
            high = row[j+1] - 1 if j < k else 1000
            
            found = False
            # Try random candidates in range
            if low < high:
                # Bias towards smaller values to reduce max
                # But we are replacing a middle element, so it doesn't affect max directly
                # unless it forces the last element to be larger.
                # Actually, replacing a middle element changes all differences.
                
                candidates = list(range(low, high + 1))
                random.shuffle(candidates)
                # Limit candidates to speed up
                candidates = candidates[:50]
                
                for new_val in candidates:
                    new_diffs = set()
                    valid = True
                    for i in range(len(row)):
                        if i == j:
                            continue
                        d = abs(new_val - row[i])
                        if d in used_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        row[j] = new_val
                        used_diffs |= new_diffs
                        found = True
                        break
            
            if not found:
                # Revert
                row[j] = old_val
                used_diffs |= to_remove
                continue

        elif move_type == 'rebuild':
            idx = random.randint(0, n_rows - 1)
            row = rows[idx]
            to_remove = set()
            for i in range(len(row)):
                for l in range(i+1, len(row)):
                    to_remove.add(row[l] - row[i])
            
            used_diffs -= to_remove
            
            new_row, new_diffs = get_greedy_row(used_diffs)
            
            if new_row is None:
                used_diffs |= to_remove
                continue
                
            rows[idx] = new_row
            used_diffs |= new_diffs

        else: # swap_diffs (complex, skip for now, rely on replace/rebuild)
            continue

        # Local improvement
        local_improve(rows, used_diffs)
        
        new_score = get_max_val(rows)
        
        # Acceptance
        delta = new_score - current_score
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_rows = [r[:] for r in rows]
        else:
            restore(rows, used_diffs, snap_rows, snap_diffs)
        
        T *= cooling_rate

    return best_rows

# Helper functions defined inside construct to keep namespace clean or defined outside?
# Defined outside for clarity in the final block.

def get_greedy_row(used_diffs, k):
    """
    Constructs a single row greedily.
    """
    row = [0]
    diffs = set()
    x = 1
    # To ensure we don't loop forever if no solution exists (shouldn't happen for small k)
    max_x = 1000 
    while len(row) < k + 1 and x < max_x:
        new_diffs = {x - a for a in row}
        if not new_diffs & used_diffs:
            row.append(x)
            diffs.update(new_diffs)
            used_diffs.update(new_diffs)
            x += 1
        else:
            x += 1
    if len(row) < k + 1:
        return None, set()
    return row, diffs

def local_improve(rows, used_diffs):
    """
    Attempts to reduce the maximum entry of each row.
    """
    improved = True
    while improved:
        improved = False
        for i in range(len(rows)):
            row = rows[i]
            if len(row) < 2:
                continue
            # Try to shrink the last element
            last_val = row[-1]
            prev_val = row[-2]
            
            # Remove differences involving last_val
            to_remove = {last_val - a for a in row[:-1]}
            used_diffs -= to_remove
            
            found = False
            # Try values from prev_val + 1 to last_val - 1
            # We want the smallest possible valid value
            for new_val in range(prev_val + 1, last_val):
                new_diffs = {new_val - a for a in row[:-1]}
                if not new_diffs & used_diffs:
                    row[-1] = new_val
                    used_diffs |= new_diffs
                    improved = True
                    found = True
                    break
            
            if not found:
                # Revert
                used_diffs |= to_remove

def get_max_val(rows):
    return max(max(row) for row in rows)

# Re-define construct to use the helpers defined above
# Wait, I defined construct inside the thought process. I need to put the final code in the block.
# The helpers get_greedy_row, local_improve, get_max_val are now global.
# I need to make sure construct uses them.

def construct(n_rows, k):
    start_time = time.time()
    timeout = 145.0
    
    # 1. Initial Solution
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row, _ = get_greedy_row(used_diffs, k)
        if row is None:
            # Fallback: just return whatever we have? 
            # Should not happen.
            return rows
        rows.append(row)
    
    local_improve(rows, used_diffs)
    
    best_rows = [r[:] for r in rows]
    best_score = get_max_val(rows)
    current_score = best_score
    
    # 2. Simulated Annealing
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.1
    
    while time.time() - start_time < timeout:
        if T < min_T:
            break
            
        # Snapshot
        snap_rows = [r[:] for r in rows]
        snap_diffs = set(used_diffs)
        
        move_type = random.choice(['replace', 'rebuild'])
        
        if move_type == 'replace':
            idx = random.randint(0, n_rows - 1)
            j = random.randint(1, k)
            row = rows[idx]
            old_val = row[j]
            
            to_remove = set()
            for i in range(len(row)):
                if i != j:
                    to_remove.add(abs(row[j] - row[i]))
            
            used_diffs -= to_remove
            
            low = row[j-1] + 1
            high = row[j+1] - 1 if j < k else 1000
            
            found = False
            if low < high:
                candidates = list(range(low, high + 1))
                random.shuffle(candidates)
                # Limit search
                candidates = candidates[:100]
                
                for new_val in candidates:
                    new_diffs = set()
                    valid = True
                    for i in range(len(row)):
                        if i == j:
                            continue
                        d = abs(new_val - row[i])
                        if d in used_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        row[j] = new_val
                        used_diffs |= new_diffs
                        found = True
                        break
            
            if not found:
                row[j] = old_val
                used_diffs |= to_remove
                continue

        else: # rebuild
            idx = random.randint(0, n_rows - 1)
            row = rows[idx]
            to_remove = set()
            for i in range(len(row)):
                for l in range(i+1, len(row)):
                    to_remove.add(row[l] - row[i])
            
            used_diffs -= to_remove
            
            new_row, new_diffs = get_greedy_row(used_diffs, k)
            
            if new_row is None:
                used_diffs |= to_remove
                continue
                
            rows[idx] = new_row
            used_diffs |= new_diffs

        # Local improvement
        local_improve(rows, used_diffs)
        
        new_score = get_max_val(rows)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_rows = [r[:] for r in rows]
        else:
            # Restore
            for i in range(n_rows):
                rows[i] = snap_rows[i][:]
            used_diffs.clear()
            used_diffs.update(snap_diffs)
        
        T *= cooling_rate

    return best_rows
