import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit (10s buffer)
    
    # Total number of differences that must be distinct
    # Each row has k+1 elements. Number of pairs is C(k+1, 2) = k(k+1)/2.
    # Total diffs = n_rows * k * (k+1) / 2
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on scope: M(M+1)/2 >= total_diffs
    lower_bound = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)
    
    best_rows = None
    best_scope = float('inf')
    
    def get_scope(rows):
        if not rows:
            return float('inf')
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
        
    def compute_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
        
    def greedy_construct(use_random=False, max_window=100):
        """
        Greedy construction: for each row, pick the smallest valid next element.
        If use_random is True, it picks randomly from a small window of valid candidates.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            success = True
            while len(row) < k + 1:
                if use_random:
                    candidates = []
                    # Look ahead in a window
                    start_x = current_max + 1
                    end_x = current_max + max_window
                    for x in range(start_x, end_x + 1):
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append(x)
                    
                    if not candidates:
                        # If no candidates in window, expand or fail
                        # Try expanding window significantly
                        end_x = current_max + 500
                        for x in range(start_x, end_x + 1):
                            new_diffs = {x - a for a in row}
                            if not (new_diffs & used):
                                candidates.append(x)
                                if len(candidates) > 20: break
                            
                        if not candidates:
                            success = False
                            break
                            
                    # Pick randomly with bias towards smaller values
                    # Weight by 1/x to prefer smaller? Or just uniform?
                    # Uniform is safer to explore space.
                    next_x = random.choice(candidates)
                else:
                    # Deterministic greedy: smallest valid x
                    next_x = current_max + 1
                    while True:
                        new_diffs = {next_x - a for a in row}
                        if not (new_diffs & used):
                            break
                        next_x += 1
                        # Safety break if we go too far? 
                        # For greedy, we just assume it will find one if scope is large enough.
                    
                row.append(next_x)
                for a in row[:-1]:
                    used.add(next_x - a)
                current_max = next_x
            
            if not success:
                return None
            rows.append(row)
        return rows

    # Phase 1: Pure Greedy
    greedy_rows = greedy_construct(use_random=False)
    if greedy_rows and is_valid(greedy_rows):
        greedy_scope = get_scope(greedy_rows)
        best_rows = greedy_rows
        best_scope = greedy_scope

    # Phase 2: Randomized Greedy with restarts
    # This explores different "shapes" of the difference sets
    num_restarts = 0
    while time.time() < deadline - 30:  # Leave 30s for local search
        rows = greedy_construct(use_random=True, max_window=50)
        if rows and is_valid(rows):
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= lower_bound:
                    return best_rows
        num_restarts += 1
        
        # Occasionally try a different window size
        if num_restarts % 10 == 0:
            greedy_rows = greedy_construct(use_random=False)
            if greedy_rows:
                s = get_scope(greedy_rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = [r[:] for r in greedy_rows]

    # Phase 3: Local Search with Row-Replacement and Element-Shift moves
    if best_rows is not None and time.time() < deadline - 2:
        current_rows = [r[:] for r in best_rows]
        current_scope = best_scope
        current_diffs = compute_diffs(current_rows)
        
        temperature = max(1.0, (current_scope - lower_bound) / 5.0)
        cooling_rate = 0.99
        
        steps = 0
        
        while time.time() < deadline - 2:
            move_type = random.random()
            modified_rows = None
            modified_diffs = None
            new_scope = float('inf')
            
            if move_type < 0.5:
                # Move 1: Shift a single element in a row
                row_idx = random.randint(0, len(current_rows) - 1)
                row = current_rows[row_idx]
                if len(row) > 1:
                    elem_idx = random.randint(1, len(row) - 1)
                    # Try shifting by small amounts
                    # We need to check validity of new differences
                    original_val = row[elem_idx]
                    
                    # Define a range of deltas to try
                    deltas = list(range(-20, 21))
                    random.shuffle(deltas)
                    
                    for delta in deltas:
                        new_val = original_val + delta
                        # Check bounds within the row
                        if new_val <= row[elem_idx - 1]:
                            continue
                        if elem_idx < len(row) - 1 and new_val >= row[elem_idx + 1]:
                            continue
                        
                        # Calculate old and new differences for this element
                        old_diffs_elem = set()
                        new_diffs_elem = set()
                        for j in range(elem_idx):
                            old_diffs_elem.add(original_val - row[j])
                            new_diffs_elem.add(new_val - row[j])
                        for j in range(elem_idx + 1, len(row)):
                            old_diffs_elem.add(row[j] - original_val)
                            new_diffs_elem.add(row[j] - new_val)
                        
                        # Check against global used set
                        # We need to remove old diffs, add new ones, and ensure no collision
                        # Note: current_diffs contains ALL diffs.
                        # It is faster to check:
                        # Are any new_diffs_elem in (current_diffs - old_diffs_elem)?
                        other_diffs = current_diffs - old_diffs_elem
                        
                        if new_diffs_elem & other_diffs:
                            continue
                        
                        # Valid move!
                        # Construct new rows and diffs
                        modified_rows = [r[:] for r in current_rows]
                        modified_rows[row_idx][elem_idx] = new_val
                        modified_diffs = (current_diffs - old_diffs_elem) | new_diffs_elem
                        
                        new_scope = get_scope(modified_rows)
                        
                        # Acceptance
                        if new_scope < current_scope:
                            current_rows = modified_rows
                            current_diffs = modified_diffs
                            current_scope = new_scope
                            
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best_rows = [r[:] for r in current_rows]
                                if best_scope <= lower_bound:
                                    return best_rows
                        else:
                            # Simulated Annealing acceptance
                            delta_E = new_scope - current_scope
                            if delta_E < 0 or random.random() < math.exp(-delta_E / temperature):
                                current_rows = modified_rows
                                current_diffs = modified_diffs
                                current_scope = new_scope
                        break # Accept first valid delta or stop trying? 
                        # Usually better to try a few, but break for speed.
                        # Let's try all deltas to find the best one? No, too slow.
                        # Just take the first valid one found in the shuffled list.
                        
            else:
                # Move 2: Replace an entire row with a new greedy row
                row_idx = random.randint(0, len(current_rows) - 1)
                
                # Calculate diffs of other rows
                other_rows = [r for i, r in enumerate(current_rows) if i != row_idx]
                other_diffs = set()
                for r in other_rows:
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            other_diffs.add(r[j] - r[i])
                
                # Try to build a new row that avoids other_diffs
                # We want to minimize the max element of this new row.
                # Greedy approach for the new row:
                new_row = [0]
                current_max = 0
                success = True
                # Limit the search window for the new row to keep it fast
                # We can use a window relative to current_max
                max_window_for_row = 200
                
                while len(new_row) < k + 1:
                    candidates = []
                    start_x = current_max + 1
                    end_x = current_max + max_window_for_row
                    
                    for x in range(start_x, end_x + 1):
                        new_diffs = {x - a for a in new_row}
                        if not (new_diffs & other_diffs):
                            candidates.append(x)
                            if len(candidates) > 10:
                                break # Take first 10 to keep it diverse but fast
                    
                    if not candidates:
                        # Expand window
                        end_x = current_max + 1000
                        for x in range(start_x, end_x + 1):
                            new_diffs = {x - a for a in new_row}
                            if not (new_diffs & other_diffs):
                                candidates.append(x)
                                if len(candidates) > 10:
                                    break
                        
                        if not candidates:
                            success = False
                            break
                    
                    # Pick the smallest candidate to minimize scope of this row?
                    # Or random? Random helps escape local minima in the global scope.
                    # Let's pick the smallest to keep the row tight, 
                    # but the global SA handles the rest.
                    next_x = candidates[0]
                    
                    new_row.append(next_x)
                    # Update local used diffs? No, we only care about global collision at the end.
                    # But we need to check internal collisions?
                    # The condition `not (new_diffs & other_diffs)` ensures no collision with OTHER rows.
                    # We must ALSO ensure no collision WITHIN the new row.
                    # The greedy step `new_diffs = {x - a for a in new_row}` checks against previous elements of the SAME row.
                    # So yes, this logic is correct.
                    
                    current_max = next_x
                
                if success and len(new_row) == k + 1:
                    # Check if this new row is valid (it should be by construction)
                    # And if it improves the scope
                    new_rows = [r[:] for r in current_rows]
                    new_rows[row_idx] = new_row
                    new_scope = get_scope(new_rows)
                    
                    # Recalculate diffs
                    old_row_diffs = set()
                    for i in range(len(current_rows[row_idx])):
                        for j in range(i + 1, len(current_rows[row_idx])):
                            old_row_diffs.add(current_rows[row_idx][j] - current_rows[row_idx][i])
                    
                    new_row_diffs = set()
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            new_row_diffs.add(new_row[j] - new_row[i])
                    
                    # Verify no collision with other rows (sanity check)
                    if not (new_row_diffs & other_diffs):
                        modified_diffs = (current_diffs - old_row_diffs) | new_row_diffs
                        modified_rows = new_rows
                        
                        if new_scope < current_scope:
                            current_rows = modified_rows
                            current_diffs = modified_diffs
                            current_scope = new_scope
                            
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best_rows = [r[:] for r in current_rows]
                                if best_scope <= lower_bound:
                                    return best_rows
                        else:
                            delta_E = new_scope - current_scope
                            if delta_E < 0 or random.random() < math.exp(-delta_E / temperature):
                                current_rows = modified_rows
                                current_diffs = modified_diffs
                                current_scope = new_scope

            temperature *= cooling_rate
            steps += 1
            
            # Occasionally restart from best found to avoid drift?
            # Or from a fresh greedy?
            if steps % 1000 == 0:
                # Cool down more
                temperature *= 0.9
                # Try to re-greedy the worst row?
                # The "Replace row" move does this.

    return best_rows
