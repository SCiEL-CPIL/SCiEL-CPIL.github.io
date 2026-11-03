import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs

    def get_scope(rows):
        return max(max(r) for r in rows)

    def compute_used_set(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def validate(rows):
        """Check if rows form a valid DTS."""
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                if i > 0 and row[i] <= row[i-1]:
                    return False
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_construct(seed, max_scope=None):
        """
        Fast greedy construction with optional scope limit.
        Uses randomized candidate ordering to explore different local optima.
        """
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                
                # Determine search range
                if max_scope is not None:
                    upper = max_scope + 1
                else:
                    upper = last + 300
                    
                # Strategy: try small increments in random order first, then linear scan
                small_range = min(30, upper - last - 1)
                if small_range > 0:
                    small_cands = list(range(last + 1, last + small_range + 1))
                    rng.shuffle(small_cands)
                    for x in small_cands:
                        new_diffs = [x - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                
                # If not found, try wider range with linear scan
                if not found:
                    for x in range(last + 1, upper):
                        new_diffs = [x - a for a in row]
                        conflict = False
                        for d in new_diffs:
                            if d in used:
                                conflict = True
                                break
                        if conflict:
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
    
    # Try many greedy seeds to get a good starting point
    for i in range(100):
        if time.time() > deadline - 60:
            break
        rows = greedy_construct(i)
        if rows:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                
    if best_rows is None:
        # Fallback: should rarely happen
        return [[0] * (k + 1) for _ in range(n_rows)]

    current_rows = [list(r) for r in best_rows]
    current_scope = best_scope
    used_set = compute_used_set(current_rows)

    # 2. Compaction-based Local Search
    def try_compact(rows, used_set):
        """
        Attempts to reduce the scope of rows by moving elements to smaller values.
        Returns (changed, used_set).
        """
        changed = False
        for r_idx in range(n_rows):
            row = rows[r_idx]
            # Process indices from last to first (excluding 0) to allow left-shifting
            for e_idx in range(len(row) - 1, 0, -1):
                old_val = row[e_idx]
                min_val = row[e_idx - 1] + 1
                
                if old_val <= min_val:
                    continue
                
                # We want to find the smallest v in [min_val, old_val) that is valid
                found_valid = False
                for new_val in range(min_val, old_val):
                    # Check bounds with next element
                    if e_idx < len(row) - 1 and new_val >= row[e_idx + 1]:
                        break
                    
                    # Calculate diffs
                    removed_diffs = set()
                    added_diffs = set()
                    for i in range(len(row)):
                        if i == e_idx:
                            continue
                        removed_diffs.add(abs(row[i] - old_val))
                        added_diffs.add(abs(row[i] - new_val))
                    
                    # Check internal duplicates in added_diffs
                    if len(added_diffs) != len(row) - 1:
                        continue
                        
                    # Check against global used set
                    conflict = False
                    for d in added_diffs:
                        if d in used_set:
                            if d not in removed_diffs:
                                conflict = True
                                break
                    if conflict:
                        continue
                        
                    # Valid!
                    row[e_idx] = new_val
                    used_set = (used_set - removed_diffs) | added_diffs
                    changed = True
                    found_valid = True
                    break
                
        return changed, used_set

    # 3. Perturbation
    def perturb(rows, used_set, rng):
        """
        Randomly increases a random element to a valid value to escape local minima.
        """
        r_idx = rng.randint(0, n_rows - 1)
        e_idx = rng.randint(1, len(rows[r_idx]) - 1)
        old_val = rows[r_idx][e_idx]
        
        max_search = old_val + 80
        
        for _ in range(20):
            new_val = rng.randint(old_val + 1, max_search)
            
            if e_idx < len(rows[r_idx]) - 1 and new_val >= rows[r_idx][e_idx + 1]:
                continue
            
            removed_diffs = set()
            added_diffs = set()
            for i in range(len(rows[r_idx])):
                if i == e_idx:
                    continue
                removed_diffs.add(abs(rows[r_idx][i] - old_val))
                added_diffs.add(abs(rows[r_idx][i] - new_val))
            
            if len(added_diffs) != len(rows[r_idx]) - 1:
                continue
                
            conflict = False
            for d in added_diffs:
                if d in used_set:
                    if d not in removed_diffs:
                        conflict = True
                        break
            if conflict:
                continue
                
            rows[r_idx][e_idx] = new_val
            used_set = (used_set - removed_diffs) | added_diffs
            return True, used_set
            
        return False, used_set

    # 4. Backtracking Search with Bitset Optimization
    def backtracking_search(target_scope, seed, deadline):
        """
        Attempts to construct a solution where all elements are <= target_scope.
        Uses randomized backtracking with bitset for difference tracking.
        """
        rng = random.Random(seed)
        # Use a set for used differences (simpler than bitset for Python)
        used = set()
        rows = []
        
        def build_row(pos, current_mark, row_so_far):
            if time.time() > deadline:
                return None
            
            if pos == k + 1:
                return row_so_far
            
            if current_mark + 1 > target_scope:
                return None
            
            # Collect valid candidates
            candidates = []
            for x in range(current_mark + 1, target_scope + 1):
                new_diffs = {x - a for a in row_so_far}
                if new_diffs & used:
                    continue
                candidates.append(x)
            
            if not candidates:
                return None
            
            # Shuffle a portion of the candidates to randomize the search path
            # Try smaller candidates first but with some randomization
            sample_pool = candidates[:30]
            rng.shuffle(sample_pool)
            
            for x in sample_pool:
                new_diffs = {x - a for a in row_so_far}
                
                # Add to used
                used.update(new_diffs)
                
                # Recurse
                res = build_row(pos + 1, x, row_so_far + [x])
                if res is not None:
                    return res
                
                # Undo
                used.difference_update(new_diffs)
                
            return None

        for i in range(n_rows):
            row = build_row(1, 0, [0])
            if row is None:
                return None
            rows.append(row)
            
        return rows

    # Main Loop
    rng = random.Random(42)
    
    # Phase A: Local Search (Compaction + Perturbation)
    iterations = 0
    max_ls_iterations = 2000
    no_improve_count = 0
    
    while time.time() < deadline - 40 and iterations < max_ls_iterations:
        iterations += 1
        
        # Compact
        changed, used_set = try_compact(current_rows, used_set)
        
        if changed:
            new_scope = get_scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        # If stuck, perturb
        if no_improve_count > 10:
            perturbed, used_set = perturb(current_rows, used_set, rng)
            if perturbed:
                # After perturbation, compact again immediately
                changed, used_set = try_compact(current_rows, used_set)
                if changed:
                    new_scope = get_scope(current_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in current_rows]
                        no_improve_count = 0
            no_improve_count = 0
            
    # Phase B: Backtracking Search for Lower Scope
    current_target = best_scope - 1
    bt_attempts = 0
    
    while time.time() < deadline - 10 and current_target > lower_bound:
        bt_attempts += 1
        
        found = False
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            
            solution = backtracking_search(current_target, seed, deadline)
            if solution:
                if validate(solution):
                    new_scope = get_scope(solution)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in solution]
                        found = True
                        # Update current_rows for further local search
                        current_rows = [list(r) for r in best_rows]
                        used_set = compute_used_set(current_rows)
                        
                        if best_scope <= lower_bound:
                            return best_rows
                        break
        
        if found:
            current_target = best_scope - 1
        else:
            current_target -= 1

    # Final Polish: Run local search again on the best solution found
    if time.time() < deadline - 3:
        current_rows = [list(r) for r in best_rows]
        used_set = compute_used_set(current_rows)
        
        for _ in range(500):
            if time.time() > deadline - 3:
                break
            changed, used_set = try_compact(current_rows, used_set)
            if changed:
                new_scope = get_scope(current_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]

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
