import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Lower bound: need n_rows * k * (k+1)/2 distinct differences in [1, M-1]
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1
    
    best_rows = None
    best_scope = float('inf')
    
    def get_used_diffs(rows):
        """Compute the set of all differences from rows."""
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used
    
    def is_valid(rows, used_set):
        """Check if all differences are distinct across all rows."""
        # Rebuild used set and check for duplicates
        seen = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True
    
    def greedy_baseline(seed_val):
        """Quick greedy to get a valid starting solution."""
        random.seed(seed_val)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                # Try small increments first
                found = False
                for x in range(last + 1, last + 50):
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                if not found:
                    # Wider search
                    for x in range(last + 50, last + 200):
                        valid = True
                        new_diffs = []
                        for a in row:
                            d = x - a
                            if d in used:
                                valid = False
                                break
                            new_diffs.append(d)
                        if valid:
                            row.append(x)
                            used.update(new_diffs)
                            found = True
                            break
                if not found:
                    return None
            rows.append(row)
        return rows
    
    def simulated_annealing(rows, used_set, max_time):
        """Run simulated annealing to minimize the scope."""
        nonlocal best_scope, best_rows
        
        rows = [list(r) for r in rows]
        used_set = set(used_set)
        scope = max(max(r) for r in rows)
        
        # SA parameters
        T = 10.0
        cooling = 0.995
        T_min = 0.01
        
        iteration = 0
        last_improve = 0
        stall_count = 0
        
        while time.time() < deadline - 5 and T > T_min:
            iteration += 1
            
            # Choose a random move type
            move_type = random.random()
            
            if move_type < 0.6:
                # Move 1: Shift a random element in a random row to a new position
                row_idx = random.randint(0, n_rows - 1)
                row = rows[row_idx]
                if len(row) < 2:
                    continue
                # Pick a non-first element
                elem_idx = random.randint(1, len(row) - 1)
                old_val = row[elem_idx]
                
                # Remove from row and used_set
                row.pop(elem_idx)
                for a in row:
                    used_set.discard(abs(old_val - a))
                
                # Try to insert at a new position
                # Prefer positions that reduce scope
                max_allowed = scope - 1 if scope > lb else scope
                candidates = []
                # Try positions near the old value and smaller
                for x in range(1, min(old_val + 5, max_allowed + 1)):
                    if x not in row:
                        # Check validity
                        valid = True
                        for a in row:
                            d = abs(x - a)
                            if d == 0 or d in used_set:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                
                if not candidates:
                    # Put back
                    pos = 0
                    while pos < len(row) and row[pos] < old_val:
                        pos += 1
                    row.insert(pos, old_val)
                    for a in row:
                        if a != old_val:
                            used_set.add(abs(old_val - a))
                    continue
                
                # Choose a candidate (bias toward smaller)
                if random.random() < 0.7:
                    new_val = min(candidates)
                else:
                    new_val = random.choice(candidates)
                
                # Insert
                pos = 0
                while pos < len(row) and row[pos] < new_val:
                    pos += 1
                row.insert(pos, new_val)
                for a in row:
                    if a != new_val:
                        used_set.add(abs(new_val - a))
                
                new_scope = max(max(r) for r in rows)
                delta = new_scope - scope
                
                if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                    scope = new_scope
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [list(r) for r in rows]
                        last_improve = iteration
                else:
                    # Revert
                    row.pop(pos)
                    for a in row:
                        used_set.discard(abs(new_val - a))
                    pos2 = 0
                    while pos2 < len(row) and row[pos2] < old_val:
                        pos2 += 1
                    row.insert(pos2, old_val)
                    for a in row:
                        if a != old_val:
                            used_set.add(abs(old_val - a))
            
            elif move_type < 0.85:
                # Move 2: Swap two elements between different rows
                if n_rows < 2:
                    continue
                row1_idx, row2_idx = random.sample(range(n_rows), 2)
                row1 = rows[row1_idx]
                row2 = rows[row2_idx]
                
                if len(row1) < 2 or len(row2) < 2:
                    continue
                
                elem1_idx = random.randint(1, len(row1) - 1)
                elem2_idx = random.randint(1, len(row2) - 1)
                val1 = row1[elem1_idx]
                val2 = row2[elem2_idx]
                
                if val1 == val2:
                    continue
                
                # Remove both
                row1.pop(elem1_idx)
                row2.pop(elem2_idx)
                
                # Remove their diffs from used_set
                for a in row1:
                    used_set.discard(abs(val1 - a))
                for a in row2:
                    used_set.discard(abs(val2 - a))
                
                # Try to insert val2 into row1 and val1 into row2
                valid1 = True
                for a in row1:
                    d = abs(val2 - a)
                    if d == 0 or d in used_set:
                        valid1 = False
                        break
                
                valid2 = True
                if valid1:
                    # Temporarily add val2's diffs
                    temp_diffs = set()
                    for a in row1:
                        d = abs(val2 - a)
                        temp_diffs.add(d)
                    for a in row2:
                        d = abs(val1 - a)
                        if d in used_set or d in temp_diffs:
                            valid2 = False
                            break
                    # Also check val1's diffs don't conflict with val2's
                    for a in row1:
                        for b in row2:
                            if abs(val2 - a) == abs(val1 - b) and abs(val2 - a) > 0:
                                # This would create duplicate if both are added
                                pass  # We'll check more carefully below
                
                if valid1 and valid2:
                    # Check for cross-conflicts more carefully
                    conflict = False
                    diffs1 = set(abs(val2 - a) for a in row1)
                    diffs2 = set(abs(val1 - a) for a in row2)
                    # Check if any diff in diffs1 conflicts with existing or diffs2
                    for d in diffs1:
                        if d in used_set or d in diffs2:
                            conflict = True
                            break
                    if not conflict:
                        for d in diffs2:
                            if d in used_set:
                                conflict = True
                                break
                
                if not (valid1 and valid2) or conflict:
                    # Revert
                    pos1 = 0
                    while pos1 < len(row1) and row1[pos1] < val1:
                        pos1 += 1
                    row1.insert(pos1, val1)
                    pos2 = 0
                    while pos2 < len(row2) and row2[pos2] < val2:
                        pos2 += 1
                    row2.insert(pos2, val2)
                    for a in row1:
                        if a != val1:
                            used_set.add(abs(val1 - a))
                    for a in row2:
                        if a != val2:
                            used_set.add(abs(val2 - a))
                    continue
                
                # Insert
                pos1 = 0
                while pos1 < len(row1) and row1[pos1] < val2:
                    pos1 += 1
                row1.insert(pos1, val2)
                pos2 = 0
                while pos2 < len(row2) and row2[pos2] < val1:
                    pos2 += 1
                row2.insert(pos2, val1)
                
                for a in row1:
                    if a != val2:
                        used_set.add(abs(val2 - a))
                for a in row2:
                    if a != val1:
                        used_set.add(abs(val1 - a))
                
                new_scope = max(max(r) for r in rows)
                delta = new_scope - scope
                
                if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                    scope = new_scope
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [list(r) for r in rows]
                        last_improve = iteration
                else:
                    # Revert
                    row1.pop(pos1)
                    row2.pop(pos2)
                    for a in row1:
                        used_set.discard(abs(val2 - a))
                    for a in row2:
                        used_set.discard(abs(val1 - a))
                    pos1 = 0
                    while pos1 < len(row1) and row1[pos1] < val1:
                        pos1 += 1
                    row1.insert(pos1, val1)
                    pos2 = 0
                    while pos2 < len(row2) and row2[pos2] < val2:
                        pos2 += 1
                    row2.insert(pos2, val2)
                    for a in row1:
                        if a != val1:
                            used_set.add(abs(val1 - a))
                    for a in row2:
                        if a != val2:
                            used_set.add(abs(val2 - a))
            
            else:
                # Move 3: Rebuild a random row from scratch (keeping 0)
                row_idx = random.randint(0, n_rows - 1)
                old_row = list(rows[row_idx])
                
                # Remove old row's diffs
                for i in range(len(old_row)):
                    for j in range(i + 1, len(old_row)):
                        used_set.discard(old_row[j] - old_row[i])
                
                # Build new row
                new_row = [0]
                row_used = set()  # Diffs used within this row
                while len(new_row) < k + 1:
                    last = new_row[-1]
                    found = False
                    # Try to find a valid next element
                    for x in range(last + 1, last + 80):
                        valid = True
                        new_diffs = []
                        for a in new_row:
                            d = x - a
                            if d in used_set or d in row_used:
                                valid = False
                                break
                            new_diffs.append(d)
                        if valid:
                            new_row.append(x)
                            row_used.update(new_diffs)
                            found = True
                            break
                    if not found:
                        # Try wider
                        for x in range(last + 80, last + 200):
                            valid = True
                            new_diffs = []
                            for a in new_row:
                                d = x - a
                                if d in used_set or d in row_used:
                                    valid = False
                                    break
                                new_diffs.append(d)
                            if valid:
                                new_row.append(x)
                                row_used.update(new_diffs)
                                found = True
                                break
                    if not found:
                        # Revert to old row
                        rows[row_idx] = old_row
                        for i in range(len(old_row)):
                            for j in range(i + 1, len(old_row)):
                                used_set.add(old_row[j] - old_row[i])
                        break
                
                if found or len(new_row) == k + 1:
                    rows[row_idx] = new_row
                    for d in row_used:
                        used_set.add(d)
                    
                    new_scope = max(max(r) for r in rows)
                    delta = new_scope - scope
                    
                    if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                        scope = new_scope
                        if scope < best_scope:
                            best_scope = scope
                            best_rows = [list(r) for r in rows]
                            last_improve = iteration
                    else:
                        # Revert
                        rows[row_idx] = old_row
                        for d in row_used:
                            used_set.discard(d)
                        for i in range(len(old_row)):
                            for j in range(i + 1, len(old_row)):
                                used_set.add(old_row[j] - old_row[i])
            
            # Cool down
            T *= cooling
            
            # Stall detection
            if iteration - last_improve > 500:
                stall_count += 1
                if stall_count > 5:
                    break
                last_improve = iteration
        
        return scope
    
    # Phase 1: Generate multiple greedy baselines
    num_baselines = min(50, int((deadline - start_time - 30) / 0.5))
    for i in range(num_baselines):
        if time.time() > deadline - 35:
            break
        rows = greedy_baseline(i * 1000 + 42)
        if rows is not None:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    # Phase 2: Run SA on the best baseline
    if best_rows is not None and time.time() < deadline - 20:
        used_set = get_used_diffs(best_rows)
        simulated_annealing(best_rows, used_set, deadline - start_time)
    
    # Phase 3: If we have time, run SA on other good baselines
    if best_rows is not None and time.time() < deadline - 15:
        # Try SA on a few more baselines
        for i in range(10):
            if time.time() > deadline - 10:
                break
            rows = greedy_baseline(100000 + i * 7777)
            if rows is not None:
                scope = max(max(r) for r in rows)
                if scope < best_scope + 20:  # Only if competitive
                    used_set = get_used_diffs(rows)
                    simulated_annealing(rows, used_set, deadline - start_time)
    
    if best_rows is None:
        # Fallback: simple greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    x += 1
                    valid = True
                    for a in row:
                        if (x - a) in used:
                            valid = False
                            break
                    if valid:
                        break
                row.append(x)
                used.update({x - a for a in row if a != x})
            best_rows.append(row)
    
    return best_rows
