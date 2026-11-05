import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Get an initial valid configuration using greedy
    initial_rows = _greedy_construct(n_rows, k)
    if not initial_rows:
        return []
    
    current_rows = [list(r) for r in initial_rows]
    current_scope = _scope(current_rows)
    
    # 2. Simulated Annealing to reduce scope
    # We want to minimize the maximum entry.
    # State: list of rows
    # Energy: max(max(r) for r in rows)
    
    # Pre-calculate difference sets for current rows for fast checking
    # diff_sets[i] is the set of differences in row i
    
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
    
    current_diffs = [get_diffs(r) for r in current_rows]
    
    # SA Parameters
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    max_iters = 200000
    
    rng = random.Random(42)
    
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    # To generate a random row, we need to pick k increasing integers.
    # To keep it fast, we restrict the max value to be around current_scope + some buffer.
    
    def generate_random_row(max_val):
        # Pick k distinct integers from 1..max_val
        # Ensure they are increasing
        candidates = list(range(1, max_val + 1))
        if len(candidates) < k:
            return None
        chosen = rng.sample(candidates, k)
        chosen.sort()
        row = [0] + chosen
        # Check internal validity (distinct diffs)
        d = get_diffs(row)
        if len(d) != len(row) * (len(row) - 1) // 2:
            return None
        return row

    def check_compatibility(new_row_diffs, other_diffs_list, exclude_idx):
        for j in range(len(other_diffs_list)):
            if j == exclude_idx:
                continue
            if new_row_diffs & other_diffs_list[j]:
                return False
        return True

    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        if T < T_min:
            T = T_min
            
        # Pick a row to modify
        idx = rng.randint(0, n_rows - 1)
        
        # Determine a reasonable max_val for the new row
        # We want to try to reduce scope, so we bias towards smaller values
        # But we also need to allow exploration
        current_row_max = max(current_rows[idx])
        
        # Try to generate a row with max value <= current_scope
        # Sometimes allow larger to escape local minima
        if rng.random() < 0.8:
            target_max = current_scope
        else:
            target_max = current_scope + 20
            
        new_row = None
        # Try a few times to generate a valid internal row
        for _ in range(10):
            new_row = generate_random_row(target_max)
            if new_row:
                break
        
        if not new_row:
            continue
            
        new_diffs = get_diffs(new_row)
        
        # Check compatibility with others
        if check_compatibility(new_diffs, current_diffs, idx):
            new_scope = max(max(r) for r in current_rows if r is not current_rows[idx])
            # Wait, calculating new_scope by iterating all rows is O(n).
            # We can maintain a count of how many rows have max == current_scope, etc.
            # But for simplicity, let's just compute it. n_rows is small.
            # Actually, we can just check if the new row's max is the new bottleneck.
            # Let's just compute max of all rows.
            # Optimization: Keep track of maxes.
            # For now, simple calculation.
            temp_rows = current_rows.copy()
            temp_rows[idx] = new_row
            new_scope_val = max(max(r) for r in temp_rows)
            
            delta = new_scope_val - current_scope
            if delta <= 0:
                # Improvement or neutral
                current_rows[idx] = new_row
                current_diffs[idx] = new_diffs
                current_scope = new_scope_val
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                # Accept with probability exp(-delta/T)
                if rng.random() < math.exp(-delta / T):
                    current_rows[idx] = new_row
                    current_diffs[idx] = new_diffs
                    current_scope = new_scope_val
        
        T *= cooling

    # 3. Final Polish: Try to reduce the max of each row greedily
    # We try to reduce the largest element of each row as much as possible
    # while maintaining validity.
    
    def _polish(rows, k, deadline):
        rows = [list(r) for r in rows]
        diffs = [get_diffs(r) for r in rows]
        scope = max(max(r) for r in rows)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(rows)):
                row = rows[i]
                # Try to reduce the last element first
                for pos in range(len(row) - 1, 0, -1):
                    old_val = row[pos]
                    # Try values from row[pos-1]+1 to old_val-1
                    for new_val in range(row[pos-1] + 1, old_val):
                        row[pos] = new_val
                        # Check internal validity
                        new_diffs = get_diffs(row)
                        if len(new_diffs) != len(row) * (len(row) - 1) // 2:
                            row[pos] = old_val
                            continue
                        # Check external compatibility
                        compatible = True
                        for j in range(len(rows)):
                            if j == i: continue
                            if new_diffs & diffs[j]:
                                compatible = False
                                break
                        if compatible:
                            diffs[i] = new_diffs
                            # Check if scope reduced
                            new_scope = max(max(r) for r in rows)
                            if new_scope < scope:
                                scope = new_scope
                                improved = True
                                break
                            else:
                                # Revert if scope didn't improve? 
                                # Actually, we want to minimize scope. 
                                # If we can reduce an element without increasing max, it might help future reductions.
                                # But for this simple polish, let's only accept if scope reduces or if it's the max row.
                                # Let's just accept if it doesn't increase scope.
                                pass
                        else:
                            row[pos] = old_val
                    if improved:
                        break
        return rows, scope

    if time.time() < deadline - 5.0:
        polished_rows, polished_scope = _polish(best_rows, k, deadline - 2.0)
        if polished_scope < best_scope:
            best_rows = polished_rows
            best_scope = polished_scope

    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(max(r) for r in rows)

def _greedy_construct(n_rows, k):
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                row.append(x)
                used_diffs.update(new_diffs)
        rows.append(row)
    return rows
