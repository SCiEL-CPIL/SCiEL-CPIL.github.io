import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Total number of differences that must be distinct
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on scope: M(M+1)/2 >= total_diffs
    lower_bound = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)
    
    best_rows = None
    best_scope = float('inf')
    
    def get_scope(rows):
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
        """Return set of all differences in rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def greedy_construct(used=None):
        """Greedy construction: for each row, pick the smallest valid next element."""
        if used is None:
            used = set()
        else:
            used = set(used)
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows, used
    
    def greedy_construct_with_prefix(prefix_rows, used):
        """Complete rows given a prefix, using greedy."""
        rows = [r[:] for r in prefix_rows]
        used = set(used)
        for _ in range(len(prefix_rows), n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows, used
    
    # Phase 1: Pure greedy
    greedy_rows, greedy_used = greedy_construct()
    greedy_scope = get_scope(greedy_rows)
    if is_valid(greedy_rows):
        best_rows = greedy_rows
        best_scope = greedy_scope
    
    # Phase 2: Randomized greedy with restarts
    num_restarts = 0
    while time.time() < deadline - 15:
        used = set()
        rows = []
        success = True
        
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Adaptive window: start small, expand if few candidates
                candidates = []
                x = current_max + 1
                window_size = max(30, k * 4)
                max_x = current_max + window_size
                
                while x <= max_x:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    # Expand window
                    window_size *= 2
                    max_x = current_max + window_size
                    x = current_max + 1
                    while x <= max_x:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append(x)
                        x += 1
                
                if not candidates:
                    success = False
                    break
                
                if len(candidates) == 1:
                    next_x = candidates[0]
                else:
                    # Weighted selection: bias towards smaller values
                    # Use a power law for stronger bias
                    weights = [(1.0 / (c - current_max)) ** 0.8 for c in candidates]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    next_x = candidates[-1]
                    for c, w in zip(candidates, weights):
                        cumulative += w
                        if r <= cumulative:
                            next_x = c
                            break
                
                row.append(next_x)
                for a in row[:-1]:
                    used.add(next_x - a)
                current_max = next_x
            
            if not success:
                break
            rows.append(row)
        
        if success and is_valid(rows):
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= lower_bound:
                    return best_rows
        
        num_restarts += 1
    
    # Phase 3: Local search with row-replacement moves
    if best_rows is not None and time.time() < deadline - 3:
        current_rows = [r[:] for r in best_rows]
        current_scope = best_scope
        temperature = max(1.0, (current_scope - lower_bound) / 5.0)
        cooling_rate = 0.998
        
        current_diffs = compute_diffs(current_rows)
        
        def try_modify(rows, diffs):
            """Try a random modification and return (modified_rows, modified_diffs) if valid, else (None, None)."""
            row_idx = random.randint(0, len(rows) - 1)
            row = rows[row_idx]
            
            # Choose move type
            move_type = random.random()
            
            if move_type < 0.5:
                # Move type 1: Shift one element in the row
                if len(row) > 1:
                    idx = random.randint(1, len(row) - 1)
                    # Try small deltas first, then larger
                    max_delta = min(20, row[idx] - row[idx - 1] - 1 if idx > 0 else 20)
                    if idx < len(row) - 1:
                        max_delta = min(max_delta, row[idx + 1] - row[idx] - 1)
                    if max_delta <= 0:
                        return None, None
                    
                    deltas = list(range(-max_delta, max_delta + 1))
                    random.shuffle(deltas)
                    for delta in deltas:
                        new_val = row[idx] + delta
                        if new_val <= row[idx - 1]:
                            continue
                        if idx < len(row) - 1 and new_val >= row[idx + 1]:
                            continue
                        
                        old_diffs = set()
                        new_diffs = set()
                        for j in range(idx):
                            old_diffs.add(row[idx] - row[j])
                            new_diffs.add(new_val - row[j])
                        for j in range(idx + 1, len(row)):
                            old_diffs.add(row[j] - row[idx])
                            new_diffs.add(row[j] - new_val)
                        
                        other_diffs = diffs - old_diffs
                        if new_diffs & other_diffs:
                            continue
                        
                        new_rows = [r[:] for r in rows]
                        new_rows[row_idx][idx] = new_val
                        new_diffs_set = diffs - old_diffs | new_diffs
                        
                        return new_rows, new_diffs_set
            
            elif move_type < 0.7:
                # Move type 2: Replace an entire row with a fresh greedy row
                other_rows = [r for i, r in enumerate(rows) if i != row_idx]
                other_diffs = set()
                for r in other_rows:
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            other_diffs.add(r[j] - r[i])
                
                # Try multiple random seeds for the new row
                best_new_row = None
                best_new_row_diffs = None
                
                for attempt in range(10):
                    new_row = [0]
                    current_max = 0
                    success_row = True
                    
                    while len(new_row) < k + 1:
                        candidates = []
                        x = current_max + 1
                        max_x = current_max + max(40, k * 5)
                        while x <= max_x:
                            new_diffs = {x - a for a in new_row}
                            if not (new_diffs & other_diffs):
                                candidates.append(x)
                            x += 1
                        
                        if not candidates:
                            # Try larger window
                            max_x = current_max + max(80, k * 10)
                            x = current_max + 1
                            while x <= max_x:
                                new_diffs = {x - a for a in new_row}
                                if not (new_diffs & other_diffs):
                                    candidates.append(x)
                                x += 1
                        
                        if not candidates:
                            success_row = False
                            break
                        
                        # Random pick with bias to smaller
                        if len(candidates) == 1:
                            next_x = candidates[0]
                        else:
                            weights = [(1.0 / (c - current_max)) ** 0.7 for c in candidates]
                            total_w = sum(weights)
                            r = random.random() * total_w
                            cumulative = 0
                            next_x = candidates[-1]
                            for c, w in zip(candidates, weights):
                                cumulative += w
                                if r <= cumulative:
                                    next_x = c
                                    break
                        
                        new_row.append(next_x)
                        current_max = next_x
                    
                    if success_row and len(new_row) == k + 1:
                        new_row_diffs = set()
                        for i in range(len(new_row)):
                            for j in range(i + 1, len(new_row)):
                                new_row_diffs.add(new_row[j] - new_row[i])
                        
                        if not (new_row_diffs & other_diffs):
                            if best_new_row is None or max(new_row) < max(best_new_row):
                                best_new_row = new_row
                                best_new_row_diffs = new_row_diffs
                
                if best_new_row is not None:
                    old_row_diffs = set()
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            old_row_diffs.add(row[j] - row[i])
                    
                    new_rows = [r[:] for r in rows]
                    new_rows[row_idx] = best_new_row
                    new_diffs_set = (diffs - old_row_diffs) | best_new_row_diffs
                    
                    return new_rows, new_diffs_set
            
            else:
                # Move type 3: Swap two elements between two different rows (if same position)
                if n_rows >= 2:
                    other_idx = random.randint(0, len(rows) - 1)
                    if other_idx == row_idx:
                        other_idx = (other_idx + 1) % len(rows)
                    
                    other_row = rows[other_idx]
                    
                    # Try swapping elements at the same position
                    if len(row) == len(other_row):
                        pos = random.randint(1, len(row) - 1)
                        # Check if swap maintains ordering
                        if (pos == 1 or row[pos] > other_row[pos - 1]) and \
                           (pos == len(row) - 1 or row[pos] < other_row[pos + 1]) and \
                           (pos == 1 or other_row[pos] > row[pos - 1]) and \
                           (pos == len(row) - 1 or other_row[pos] < row[pos + 1]):
                            
                            # Compute old and new diffs
                            old_diffs_row1 = set()
                            new_diffs_row1 = set()
                            for j in range(pos):
                                old_diffs_row1.add(row[pos] - row[j])
                                new_diffs_row1.add(other_row[pos] - row[j])
                            for j in range(pos + 1, len(row)):
                                old_diffs_row1.add(row[j] - row[pos])
                                new_diffs_row1.add(row[j] - other_row[pos])
                            
                            old_diffs_row2 = set()
                            new_diffs_row2 = set()
                            for j in range(pos):
                                old_diffs_row2.add(other_row[pos] - other_row[j])
                                new_diffs_row2.add(row[pos] - other_row[j])
                            for j in range(pos + 1, len(other_row)):
                                old_diffs_row2.add(other_row[j] - other_row[pos])
                                new_diffs_row2.add(other_row[j] - row[pos])
                            
                            old_all = old_diffs_row1 | old_diffs_row2
                            new_all = new_diffs_row1 | new_diffs_row2
                            
                            other_diffs = diffs - old_all
                            if new_all & other_diffs:
                                return None, None
                            
                            new_rows = [r[:] for r in rows]
                            new_rows[row_idx][pos] = other_row[pos]
                            new_rows[other_idx][pos] = row[pos]
                            new_diffs_set = (diffs - old_all) | new_all
                            
                            return new_rows, new_diffs_set
            
            return None, None
        
        steps = 0
        no_improve_count = 0
        while time.time() < deadline - 3:
            modified_rows, modified_diffs = try_modify(current_rows, current_diffs)
            if modified_rows is not None:
                new_scope = get_scope(modified_rows)
                if new_scope < current_scope:
                    current_rows = modified_rows
                    current_diffs = modified_diffs
                    current_scope = new_scope
                    no_improve_count = 0
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in current_rows]
                        if best_scope <= lower_bound:
                            return best_rows
                elif new_scope == current_scope:
                    # Accept equal-scope moves to explore
                    if random.random() < 0.5:
                        current_rows = modified_rows
                        current_diffs = modified_diffs
                        no_improve_count += 1
                else:
                    # Simulated annealing acceptance
                    delta = new_scope - current_scope
                    if random.random() < math.exp(-delta / max(temperature, 0.1)):
                        current_rows = modified_rows
                        current_diffs = modified_diffs
                        current_scope = new_scope
                        no_improve_count += 1
                    else:
                        no_improve_count += 1
            else:
                no_improve_count += 1
            
            temperature *= cooling_rate
            
            # If stuck, restart from best solution
            if no_improve_count > 50:
                current_rows = [r[:] for r in best_rows]
                current_diffs = compute_diffs(current_rows)
                current_scope = best_scope
                temperature = max(1.0, (current_scope - lower_bound) / 5.0)
                no_improve_count = 0
            
            steps += 1
            
            # Occasionally try a fresh greedy restart
            if steps % 300 == 0 and random.random() < 0.05:
                fresh_rows, fresh_used = greedy_construct()
                if is_valid(fresh_rows):
                    fresh_scope = get_scope(fresh_rows)
                    if fresh_scope < current_scope:
                        current_rows = fresh_rows
                        current_diffs = compute_diffs(fresh_rows)
                        current_scope = fresh_scope
                        if fresh_scope < best_scope:
                            best_scope = fresh_scope
                            best_rows = [r[:] for r in current_rows]
    
    return best_rows
