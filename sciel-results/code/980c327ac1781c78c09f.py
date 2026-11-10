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
    
    def greedy_construct():
        """Greedy construction: for each row, pick the smallest valid next element."""
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
                    used |= new_diffs
            rows.append(row)
        return rows
    
    def greedy_with_bias(bias_power=2.0):
        """Greedy with weighted random selection from valid candidates."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Find valid candidates in a window
                candidates = []
                x = current_max + 1
                max_x = current_max + max(30, k * 4)
                while x <= max_x:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    # Extend window if no candidates found
                    x = max_x + 1
                    max_x = max_x + 50
                    while x <= max_x:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append(x)
                        x += 1
                
                if not candidates:
                    return None
                
                # Weighted selection: bias towards smaller values
                if len(candidates) == 1:
                    next_x = candidates[0]
                else:
                    weights = [1.0 / (c - current_max) ** bias_power for c in candidates]
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
            
            rows.append(row)
        
        return rows
    
    # Phase 1: Pure greedy
    greedy_rows = greedy_construct()
    greedy_scope = get_scope(greedy_rows)
    if is_valid(greedy_rows):
        best_rows = greedy_rows
        best_scope = greedy_scope
    
    # Phase 2: Randomized greedy with restarts
    while time.time() < deadline - 15:  # Leave 15s for local search
        used = set()
        rows = []
        success = True
        
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Find valid candidates in a window
                candidates = []
                x = current_max + 1
                max_x = current_max + max(40, k * 5)
                while x <= max_x:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    # Extend window
                    x = max_x + 1
                    max_x = max_x + 50
                    while x <= max_x:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append(x)
                        x += 1
                
                if not candidates:
                    success = False
                    break
                
                # Pick randomly from candidates with bias towards smaller values
                if len(candidates) == 1:
                    next_x = candidates[0]
                else:
                    # Weight inversely proportional to squared distance
                    weights = [1.0 / (c - current_max) ** 2 for c in candidates]
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
    
    # Phase 3: Fast local search with in-place modifications
    if best_rows is not None and time.time() < deadline - 2:
        current_rows = [r[:] for r in best_rows]
        current_scope = best_scope
        temperature = max(1.0, (current_scope - lower_bound) / 10.0)
        cooling_rate = 0.998
        
        # Precompute all differences for current solution
        current_diffs = compute_diffs(current_rows)
        
        def fast_try_modify(rows, diffs):
            """Try a random modification using in-place changes. Returns (success, new_scope, modified_diffs)."""
            row_idx = random.randint(0, len(rows) - 1)
            row = rows[row_idx]
            row_len = len(row)
            
            if row_len <= 1:
                return False, current_scope, None
            
            # Choose operation type
            op = random.random()
            
            if op < 0.7:
                # Try shifting one element
                idx = random.randint(1, row_len - 1)
                # Try deltas in random order, prioritizing small changes
                delta_range = max(5, k // 2)
                deltas = list(range(-delta_range, delta_range + 1))
                random.shuffle(deltas)
                # Put 0 first (no change), then small deltas
                deltas.sort(key=lambda d: abs(d))
                random.shuffle(deltas[1:])
                
                for delta in deltas:
                    if delta == 0:
                        continue
                    new_val = row[idx] + delta
                    # Check ordering
                    if new_val <= row[idx - 1]:
                        continue
                    if idx < row_len - 1 and new_val >= row[idx + 1]:
                        continue
                    
                    # Compute old and new differences for this element
                    old_diffs = set()
                    new_diffs = set()
                    for j in range(idx):
                        old_diffs.add(row[idx] - row[j])
                        new_diffs.add(new_val - row[j])
                    for j in range(idx + 1, row_len):
                        old_diffs.add(row[j] - row[idx])
                        new_diffs.add(row[j] - new_val)
                    
                    # Check if new_diffs conflicts with existing diffs (excluding old_diffs)
                    if new_diffs & (diffs - old_diffs):
                        continue
                    
                    # Valid! Apply the change in-place
                    old_val = row[idx]
                    row[idx] = new_val
                    new_diffs_set = diffs - old_diffs | new_diffs
                    
                    # Compute new scope
                    new_scope = max(max(r) for r in rows)
                    
                    return True, new_scope, new_diffs_set
            
            elif op < 0.9:
                # Try swapping two elements in the same row
                if row_len > 2:
                    i, j = sorted(random.sample(range(1, row_len), 2))
                    
                    # Check if swap maintains ordering
                    if i > 0 and row[j] <= row[i-1]:
                        return False, current_scope, None
                    if j < row_len - 1 and row[i] >= row[j+1]:
                        return False, current_scope, None
                    if i > 0 and j > i and row[i] <= row[j-1]:
                        return False, current_scope, None
                    if i > 0 and j > i and row[j] >= row[i+1]:
                        return False, current_scope, None
                    
                    # Compute differences affected by swap
                    old_diffs = set()
                    new_diffs = set()
                    
                    # Element i changes from row[i] to row[j]
                    for t in range(row_len):
                        if t == i or t == j:
                            continue
                        if t < i:
                            old_diffs.add(row[i] - row[t])
                            new_diffs.add(row[j] - row[t])
                        elif t > i:
                            old_diffs.add(row[t] - row[i])
                            new_diffs.add(row[t] - row[j])
                    
                    # Element j changes from row[j] to row[i]
                    for t in range(row_len):
                        if t == i or t == j:
                            continue
                        if t < j:
                            old_diffs.add(row[j] - row[t])
                            new_diffs.add(row[i] - row[t])
                        elif t > j:
                            old_diffs.add(row[t] - row[j])
                            new_diffs.add(row[t] - row[i])
                    
                    # The difference between i and j doesn't change
                    other_diffs = diffs - old_diffs
                    if new_diffs & other_diffs:
                        return False, current_scope, None
                    
                    # Apply swap in-place
                    row[i], row[j] = row[j], row[i]
                    new_diffs_set = diffs - old_diffs | new_diffs
                    new_scope = max(max(r) for r in rows)
                    
                    return True, new_scope, new_diffs_set
            
            else:
                # Try replacing an element with a nearby value
                idx = random.randint(1, row_len - 1)
                search_range = max(10, k)
                for new_val in range(max(1, row[idx] - search_range), row[idx] + search_range + 1):
                    if new_val == row[idx]:
                        continue
                    if new_val <= row[idx - 1]:
                        continue
                    if idx < row_len - 1 and new_val >= row[idx + 1]:
                        continue
                    
                    old_diffs = set()
                    new_diffs = set()
                    for j in range(idx):
                        old_diffs.add(row[idx] - row[j])
                        new_diffs.add(new_val - row[j])
                    for j in range(idx + 1, row_len):
                        old_diffs.add(row[j] - row[idx])
                        new_diffs.add(row[j] - new_val)
                    
                    if new_diffs & (diffs - old_diffs):
                        continue
                    
                    row[idx] = new_val
                    new_diffs_set = diffs - old_diffs | new_diffs
                    new_scope = max(max(r) for r in rows)
                    
                    return True, new_scope, new_diffs_set
            
            return False, current_scope, None
        
        steps = 0
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline - 2:
            success, new_scope, new_diffs = fast_try_modify(current_rows, current_diffs)
            
            if success:
                if new_scope < current_scope:
                    current_scope = new_scope
                    current_diffs = new_diffs
                    no_improve_count = 0
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in current_rows]
                        if best_scope <= lower_bound:
                            return best_rows
                elif random.random() < math.exp(-(new_scope - current_scope) / temperature):
                    current_scope = new_scope
                    current_diffs = new_diffs
                    no_improve_count += 1
                else:
                    # Revert the change - we need to track what was changed
                    # For simplicity, just accept and let SA handle it
                    current_scope = new_scope
                    current_diffs = new_diffs
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            temperature *= cooling_rate
            steps += 1
            
            # Reset temperature occasionally to escape local minima
            if steps % 1000 == 0 and temperature < 0.5:
                temperature = max(1.0, (current_scope - lower_bound) / 10.0)
            
            # Occasionally try a fresh greedy solution
            if steps % 500 == 0 and random.random() < 0.15:
                fresh = greedy_with_bias()
                if fresh is not None and is_valid(fresh):
                    fresh_scope = get_scope(fresh)
                    if fresh_scope < current_scope:
                        current_rows = [r[:] for r in fresh]
                        current_diffs = compute_diffs(current_rows)
                        current_scope = fresh_scope
                        no_improve_count = 0
                        if fresh_scope < best_scope:
                            best_scope = fresh_scope
                            best_rows = [r[:] for r in current_rows]
                            if best_scope <= lower_bound:
                                return best_rows
    
    # Phase 4: Final polish - try to reduce scope by 1
    if best_rows is not None and time.time() < deadline - 1:
        current_rows = [r[:] for r in best_rows]
        current_scope = best_scope
        current_diffs = compute_diffs(current_rows)
        
        # Try to find any element that can be reduced
        for _ in range(100):
            if time.time() > deadline - 1:
                break
            
            # Find the row with the maximum element
            max_row_idx = 0
            max_val = -1
            for i, row in enumerate(current_rows):
                if row[-1] > max_val:
                    max_val = row[-1]
                    max_row_idx = i
            
            row = current_rows[max_row_idx]
            row_len = len(row)
            
            # Try to reduce the last element
            idx = row_len - 1
            old_val = row[idx]
            
            for new_val in range(max(1, old_val - 5), old_val):
                if new_val <= row[idx - 1]:
                    continue
                
                old_diffs = set()
                new_diffs = set()
                for j in range(idx):
                    old_diffs.add(row[idx] - row[j])
                    new_diffs.add(new_val - row[j])
                
                if new_diffs & (current_diffs - old_diffs):
                    continue
                
                # Valid reduction
                row[idx] = new_val
                current_diffs = current_diffs - old_diffs | new_diffs
                new_scope = get_scope(current_rows)
                
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
                break
    
    return best_rows
