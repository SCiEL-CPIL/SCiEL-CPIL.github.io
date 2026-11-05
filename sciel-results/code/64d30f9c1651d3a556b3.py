import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    def is_valid(rows):
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_scope(rows):
        return max(max(r) for r in rows)

    def frequency_aware_construct(scope_limit, rng=None):
        """
        Structural change: frequency-aware construction.
        Instead of just checking if a diff is used, we score candidates by
        how "expensive" they are in terms of consuming rare difference values.
        This helps preserve flexibility for later rows.
        """
        if rng is None:
            rng = random.Random(42)
        
        used_diffs = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last_val = row[-1]
                best_x = None
                best_score = float('inf')
                
                # Determine search range
                start_x = last_val + 1
                end_x = min(scope_limit, last_val + scope_limit)
                
                for x in range(start_x, end_x + 1):
                    # Compute all new differences
                    new_diffs = [x - y for y in row]
                    
                    # Check validity
                    valid = True
                    for d in new_diffs:
                        if d in used_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Score: prefer candidates that use smaller differences
                    # and don't create large gaps
                    max_d = max(new_diffs)
                    min_d = min(new_diffs)
                    
                    # Pressure: minimize max difference, break ties by minimizing sum
                    score = max_d * 1000 + sum(new_diffs)
                    
                    if score < best_score:
                        best_score = score
                        best_x = x
                        # Early exit if we found a very good candidate
                        if max_d == x - last_val:
                            break
                
                if best_x is None:
                    return None
                
                row.append(best_x)
                for y in row[:-1]:
                    used_diffs.add(best_x - y)
            
            rows.append(row)
        
        return rows

    def targeted_local_search(rows, time_budget, rng=None):
        """
        Targeted local search: focus specifically on reducing the maximum element.
        Uses a more aggressive strategy for the last position of each row.
        """
        if rng is None:
            rng = random.Random(42)
        
        rows = [r[:] for r in rows]
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        t_end = time.time() + time_budget
        no_improve = 0
        max_no_improve = 3000
        
        while time.time() < t_end and no_improve < max_no_improve:
            # Focus on the last element of a random row (most impactful for scope)
            row_idx = rng.randint(0, n_rows - 1)
            pos = k  # Always target the last position
            
            old_val = rows[row_idx][pos]
            min_val = rows[row_idx][pos - 1] + 1
            max_val = old_val - 1  # Must decrease to improve scope
            
            if min_val > max_val:
                no_improve += 1
                continue
            
            # Remove old diffs involving this position
            old_diffs = []
            for i in range(k + 1):
                if i != pos:
                    d = old_val - rows[row_idx][i]
                    old_diffs.append(d)
            for d in old_diffs:
                used_diffs.discard(d)
            
            # Try to find a smaller valid value
            best_new_val = None
            for new_val in range(min_val, max_val + 1):
                valid = True
                for i in range(k + 1):
                    if i != pos:
                        d = new_val - rows[row_idx][i]
                        if d in used_diffs:
                            valid = False
                            break
                if valid:
                    # Check for internal duplicates
                    new_diffs = [new_val - rows[row_idx][i] for i in range(k + 1) if i != pos]
                    if len(set(new_diffs)) == len(new_diffs):
                        best_new_val = new_val
                        break  # Take the smallest valid value
            
            if best_new_val is not None:
                # Apply
                for i in range(k + 1):
                    if i != pos:
                        d = best_new_val - rows[row_idx][i]
                        used_diffs.add(d)
                rows[row_idx][pos] = best_new_val
                current_scope = get_scope(rows)
                
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Restore old diffs
                for d in old_diffs:
                    used_diffs.add(d)
                no_improve += 1
        
        return best_rows, best_scope

    def hybrid_search(rows, time_budget, rng=None):
        """
        Hybrid: combine targeted last-element reduction with occasional
        full-row rebuilds to escape local minima.
        """
        if rng is None:
            rng = random.Random(42)
        
        rows = [r[:] for r in rows]
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        t_end = time.time() + time_budget
        no_improve = 0
        max_no_improve = 5000
        rebuild_counter = 0
        
        while time.time() < t_end and no_improve < max_no_improve:
            rebuild_counter += 1
            
            # Every 200 iterations, try a full row rebuild
            if rebuild_counter % 200 == 0:
                row_idx = rng.randint(0, n_rows - 1)
                old_row = rows[row_idx][:]
                
                # Remove all diffs from this row
                for i in range(len(old_row)):
                    for j in range(i + 1, len(old_row)):
                        used_diffs.discard(old_row[j] - old_row[i])
                
                # Rebuild the row using frequency-aware greedy
                new_row = [0]
                while len(new_row) < k + 1:
                    last_val = new_row[-1]
                    best_x = None
                    best_score = float('inf')
                    
                    for x in range(last_val + 1, best_scope + 1):
                        new_diffs = [x - y for y in new_row]
                        valid = True
                        for d in new_diffs:
                            if d in used_diffs:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        max_d = max(new_diffs)
                        score = max_d * 1000 + sum(new_diffs)
                        
                        if score < best_score:
                            best_score = score
                            best_x = x
                            if max_d == x - last_val:
                                break
                    
                    if best_x is None:
                        # Restore old row
                        for i in range(len(old_row)):
                            for j in range(i + 1, len(old_row)):
                                used_diffs.add(old_row[j] - old_row[i])
                        break
                    
                    new_row.append(best_x)
                    for y in new_row[:-1]:
                        used_diffs.add(best_x - y)
                
                # Check if new row is valid and complete
                if len(new_row) == k + 1:
                    new_scope = get_scope(rows[:row_idx] + [new_row] + rows[row_idx+1:])
                    if new_scope <= current_scope:
                        rows[row_idx] = new_row
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                            no_improve = 0
                        else:
                            no_improve += 1
                        continue
                else:
                    # Restore old row
                    for i in range(len(old_row)):
                        for j in range(i + 1, len(old_row)):
                            used_diffs.add(old_row[j] - old_row[i])
                    no_improve += 1
                    continue
            
            # Normal move: reduce last element
            row_idx = rng.randint(0, n_rows - 1)
            pos = k
            
            old_val = rows[row_idx][pos]
            min_val = rows[row_idx][pos - 1] + 1
            max_val = old_val - 1
            
            if min_val > max_val:
                no_improve += 1
                continue
            
            old_diffs = []
            for i in range(k + 1):
                if i != pos:
                    d = old_val - rows[row_idx][i]
                    old_diffs.append(d)
            for d in old_diffs:
                used_diffs.discard(d)
            
            best_new_val = None
            for new_val in range(min_val, max_val + 1):
                valid = True
                for i in range(k + 1):
                    if i != pos:
                        d = new_val - rows[row_idx][i]
                        if d in used_diffs:
                            valid = False
                            break
                if valid:
                    new_diffs = [new_val - rows[row_idx][i] for i in range(k + 1) if i != pos]
                    if len(set(new_diffs)) == len(new_diffs):
                        best_new_val = new_val
                        break
            
            if best_new_val is not None:
                for i in range(k + 1):
                    if i != pos:
                        d = best_new_val - rows[row_idx][i]
                        used_diffs.add(d)
                rows[row_idx][pos] = best_new_val
                current_scope = get_scope(rows)
                
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                for d in old_diffs:
                    used_diffs.add(d)
                no_improve += 1
        
        return best_rows, best_scope

    # === Main strategy ===
    
    # Step 1: Frequency-aware construction with multiple scopes
    initial_rows = None
    best_initial_scope = float('inf')
    
    for scope in [135, 140, 145, 150, 160, 180, 200, 300]:
        if time.time() > deadline - 5:
            break
        rows = frequency_aware_construct(scope, random.Random(42))
        if rows is not None and is_valid(rows):
            sc = get_scope(rows)
            if sc < best_initial_scope:
                best_initial_scope = sc
                initial_rows = rows
            if sc <= 135:
                break
    
    if initial_rows is None:
        # Fallback
        used_diffs = set()
        initial_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                valid = True
                for y in row:
                    if x - y in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for y in row[:-1]:
                        used_diffs.add(x - y)
            initial_rows.append(row)
        return initial_rows
    
    current_scope = get_scope(initial_rows)
    
    # Step 2: Targeted local search
    remaining = deadline - time.time()
    if remaining > 15:
        ls_rows, ls_scope = targeted_local_search(initial_rows, remaining * 0.3, random.Random(42))
        if ls_rows is not None and is_valid(ls_rows) and ls_scope < current_scope:
            initial_rows = ls_rows
            current_scope = ls_scope
    
    # Step 3: Hybrid search
    remaining = deadline - time.time()
    if remaining > 15:
        hs_rows, hs_scope = hybrid_search(initial_rows, remaining * 0.5, random.Random(123))
        if hs_rows is not None and is_valid(hs_rows) and hs_scope < current_scope:
            initial_rows = hs_rows
            current_scope = hs_scope
    
    # Step 4: Final polish
    remaining = deadline - time.time()
    if remaining > 10:
        final_rows, final_scope = targeted_local_search(initial_rows, remaining * 0.7, random.Random(999))
        if final_rows is not None and is_valid(final_rows) and final_scope < current_scope:
            initial_rows = final_rows
            current_scope = final_scope
    
    return initial_rows
