"""Rewritten program for difference triangle set (7,5) task."""
import time
import random
import copy

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Helper: compute all positive differences for a row
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper: check if a full set of rows is valid (all diffs distinct)
    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            for d in row_diffs(row):
                if d in all_diffs:
                    return False
                all_diffs.add(d)
        return True

    # Helper: compute scope (max entry)
    def scope(rows):
        return max(max(r) for r in rows)

    # Helper: count conflicts (number of duplicate differences)
    def count_conflicts(rows):
        from collections import Counter
        all_diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    all_diffs.append(row[j] - row[i])
        c = Counter(all_diffs)
        return sum(v - 1 for v in c.values() if v > 1)

    # Greedy construction: build rows one by one, choosing smallest next mark
    def greedy_construct(n_rows, k, start_offset=0):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Interleaved greedy: build all rows simultaneously, adding one mark at a time
    def interleaved_greedy(n_rows, k):
        used = set()
        rows = [[0] for _ in range(n_rows)]
        # We need to add k more marks to each row
        # Process in rounds; in each round, for each row, try to add the smallest valid next mark
        for round_idx in range(k):
            # For each row, find the smallest x > current_max such that new diffs don't conflict
            # To balance, process rows in order but try to keep them close
            for r_idx in range(n_rows):
                row = rows[r_idx]
                current_max = row[-1]
                x = current_max
                while True:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
        return rows

    # Local search: hill climbing with random restarts
    def local_search(initial_rows, n_rows, k, time_limit=130.0):
        best_rows = copy.deepcopy(initial_rows)
        best_scope = scope(best_rows)
        best_conflicts = count_conflicts(best_rows)
        
        current_rows = copy.deepcopy(initial_rows)
        current_scope = scope(current_rows)
        current_conflicts = count_conflicts(current_rows)
        
        start_time = time.time()
        no_improve_count = 0
        max_no_improve = 20000
        
        while time.time() < time_limit and no_improve_count < max_no_improve:
            # Choose a random row to modify
            r_idx = random.randint(0, n_rows - 1)
            old_row = list(current_rows[r_idx])
            
            # Strategy: modify the row by shifting some tail elements
            # Pick a random position to start shifting
            pos = random.randint(1, len(old_row) - 1)
            # Try shifting the tail by a small random amount
            shift = random.randint(-3, 3)
            if shift == 0:
                shift = random.choice([-1, 1])
            
            new_row = old_row[:pos] + [x + shift for x in old_row[pos:]]
            
            # Ensure strictly increasing and starts at 0
            valid = True
            if new_row[0] != 0:
                valid = False
            for i in range(1, len(new_row)):
                if new_row[i] <= new_row[i-1]:
                    valid = False
                    break
            
            if not valid:
                no_improve_count += 1
                continue
            
            # Compute new conflicts and scope
            # Remove old row's diffs, add new row's diffs
            old_diffs = row_diffs(old_row)
            new_diffs = row_diffs(new_row)
            
            # Check if new diffs conflict with other rows' diffs
            other_diffs = set()
            for i, row in enumerate(current_rows):
                if i != r_idx:
                    other_diffs |= row_diffs(row)
            
            # Also check within new row for duplicates (shouldn't happen if increasing)
            if len(new_diffs) != len(list(new_diffs)):
                no_improve_count += 1
                continue
            
            conflict_count = len(new_diffs & other_diffs)
            
            # Compute scope
            new_scope = max(max(row) for row in current_rows) if r_idx == 0 else max(current_rows[r_idx][-1], max(max(r) for i, r in enumerate(current_rows) if i != r_idx))
            # Simpler: just compute full scope
            temp_rows = list(current_rows)
            temp_rows[r_idx] = new_row
            new_scope_val = max(max(r) for r in temp_rows)
            
            # Accept if fewer conflicts, or same conflicts but smaller scope
            if conflict_count < current_conflicts:
                current_rows[r_idx] = new_row
                current_conflicts = conflict_count
                current_scope = new_scope_val
                no_improve_count = 0
                
                if current_conflicts == 0 and current_scope < best_scope:
                    best_rows = copy.deepcopy(current_rows)
                    best_scope = current_scope
                    
            elif conflict_count == current_conflicts:
                if new_scope_val < current_scope:
                    current_rows[r_idx] = new_row
                    current_scope = new_scope_val
                    no_improve_count = 0
                    
                    if current_conflicts == 0 and current_scope < best_scope:
                        best_rows = copy.deepcopy(current_rows)
                        best_scope = current_scope
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            # Occasionally try a full row replacement
            if random.random() < 0.1:
                r_idx2 = random.randint(0, n_rows - 1)
                # Generate a new greedy row for this position
                used_other = set()
                for i, row in enumerate(current_rows):
                    if i != r_idx2:
                        used_other |= row_diffs(row)
                
                new_row2 = [0]
                x = 0
                while len(new_row2) < k + 1:
                    x += 1
                    nd = {x - a for a in new_row2}
                    if not nd & used_other:
                        new_row2.append(x)
                        used_other |= nd
                
                if new_row2[-1] < current_rows[r_idx2][-1]:
                    old_diffs2 = row_diffs(current_rows[r_idx2])
                    new_diffs2 = row_diffs(new_row2)
                    other_diffs2 = set()
                    for i, row in enumerate(current_rows):
                        if i != r_idx2:
                            other_diffs2 |= row_diffs(row)
                    conf2 = len(new_diffs2 & other_diffs2)
                    if conf2 < current_conflicts:
                        current_rows[r_idx2] = new_row2
                        current_conflicts = conf2
                        current_scope = max(max(r) for r in current_rows)
                        no_improve_count = 0
                        if current_conflicts == 0 and current_scope < best_scope:
                            best_rows = copy.deepcopy(current_rows)
                            best_scope = current_scope
                    elif conf2 == current_conflicts:
                        new_scope2 = max(max(r) for r in current_rows) if r_idx2 == 0 else max(new_row2[-1], max(max(r) for i, r in enumerate(current_rows) if i != r_idx2))
                        if new_scope2 < current_scope:
                            current_rows[r_idx2] = new_row2
                            current_scope = new_scope2
                            no_improve_count = 0
                            if current_conflicts == 0 and current_scope < best_scope:
                                best_rows = copy.deepcopy(current_rows)
                                best_scope = current_scope
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1
        
        return best_rows, best_scope

    # Generate initial solutions using different strategies
    initial_solutions = []
    
    # Standard greedy
    initial_solutions.append(greedy_construct(n_rows, k))
    
    # Interleaved greedy
    initial_solutions.append(interleaved_greedy(n_rows, k))
    
    # Randomized greedy with small perturbations
    for _ in range(3):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try a few candidates, pick one randomly among the smallest valid
                candidates = []
                for trial in range(20):
                    x += 1
                    nd = {x - a for a in row}
                    if not nd & used:
                        candidates.append(x)
                        if len(candidates) >= 3:
                            break
                if candidates:
                    x = random.choice(candidates)
                    nd = {x - a for a in row}
                    row.append(x)
                    used |= nd
                else:
                    # Fallback: keep incrementing until valid
                    while True:
                        x += 1
                        nd = {x - a for a in row}
                        if not nd & used:
                            row.append(x)
                            used |= nd
                            break
            rows.append(row)
        initial_solutions.append(rows)
    
    # Run local search on each initial solution
    best_overall = None
    best_scope_overall = float('inf')
    
    for init in initial_solutions:
        if time.time() > deadline - 10:
            break
        result, sc = local_search(init, n_rows, k, time_limit=deadline - time.time() - 5)
        if sc < best_scope_overall:
            best_scope_overall = sc
            best_overall = result
    
    # If we haven't found a valid solution, try a more aggressive search
    if best_overall is None or count_conflicts(best_overall) > 0:
        # Try simulated annealing style
        if time.time() < deadline - 5:
            current = copy.deepcopy(initial_solutions[0])
            current_conflicts = count_conflicts(current)
            current_scope = scope(current)
            
            temp = 50.0
            cooling = 0.999
            
            while time.time() < deadline - 2 and temp > 0.1:
                r_idx = random.randint(0, n_rows - 1)
                old_row = list(current[r_idx])
                pos = random.randint(1, len(old_row) - 1)
                shift = random.randint(-5, 5)
                if shift == 0:
                    shift = random.choice([-1, 1])
                
                new_row = old_row[:pos] + [x + shift for x in old_row[pos:]]
                
                valid = new_row[0] == 0 and all(new_row[i] > new_row[i-1] for i in range(1, len(new_row)))
                if not valid:
                    temp *= cooling
                    continue
                
                other_diffs = set()
                for i, row in enumerate(current):
                    if i != r_idx:
                        other_diffs |= row_diffs(row)
                
                new_diffs = row_diffs(new_row)
                conflict_count = len(new_diffs & other_diffs)
                
                temp_rows = list(current)
                temp_rows[r_idx] = new_row
                new_scope_val = max(max(r) for r in temp_rows)
                
                delta_conf = conflict_count - current_conflicts
                delta_scope = new_scope_val - current_scope
                delta = delta_conf * 10 + delta_scope
                
                if delta < 0 or random.random() < pow(2.718281828, -delta / temp):
                    current[r_idx] = new_row
                    current_conflicts = conflict_count
                    current_scope = new_scope_val
                    
                    if current_conflicts == 0 and current_scope < best_scope_overall:
                        best_scope_overall = current_scope
                        best_overall = copy.deepcopy(current)
                
                temp *= cooling
            
            if best_overall is None:
                best_overall = current
                best_scope_overall = scope(best_overall)
    
    # Final fallback: return the best we have
    if best_overall is None:
        best_overall = greedy_construct(n_rows, k)
    
    return best_overall
# EVOLVE-BLOCK-END
