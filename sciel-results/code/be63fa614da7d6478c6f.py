import random
import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Helper: compute all positive differences for a row
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Helper: check if a set of rows is valid (all diffs distinct across rows)
    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            rd = row_diffs(row)
            if rd & all_diffs:
                return False
            all_diffs |= rd
        return True
    
    # Helper: compute scope (max entry)
    def scope(rows):
        return max(max(row) for row in rows)
    
    # Helper: count conflicts (number of duplicate differences)
    def count_conflicts(rows):
        from collections import Counter
        all_diffs = []
        for row in rows:
            rd = row_diffs(row)
            all_diffs.extend(rd)
        c = Counter(all_diffs)
        return sum(v - 1 for v in c.values() if v > 1)
    
    # Initial solution: greedy construction
    def greedy_init():
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
    
    best_rows = greedy_init()
    best_scope = scope(best_rows)
    best_conflicts = count_conflicts(best_rows)
    
    if best_conflicts == 0:
        return best_rows
    
    # Simulated annealing with row-reconstruction moves
    rng = random.Random(42)
    
    # Temperature schedule
    T = 1000.0
    cooling = 0.995
    
    # Precompute: for a given row, what diffs does it use?
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Move: replace row r with a new random increasing sequence starting at 0
    def random_row(max_val):
        # Generate k+1 increasing integers starting at 0, all <= max_val
        # Use random subset of {1, 2, ..., max_val} of size k
        if max_val < k:
            return None
        candidates = list(range(1, max_val + 1))
        chosen = rng.sample(candidates, k)
        chosen.sort()
        return [0] + chosen
    
    # Evaluate a candidate row given the other rows' diffs
    def eval_row_candidate(candidate_row, other_diffs):
        rd = get_row_diffs(candidate_row)
        conflicts = len(rd & other_diffs)
        # Also penalize large scope
        return conflicts, max(candidate_row)
    
    # Main SA loop
    current_rows = [list(r) for r in best_rows]
    current_conflicts = best_conflicts
    current_scope = best_scope
    
    # Precompute diffs for each row
    row_diffs_list = [get_row_diffs(r) for r in current_rows]
    
    # Global diff counter for quick conflict counting
    from collections import Counter
    global_diff_counter = Counter()
    for rd in row_diffs_list:
        for d in rd:
            global_diff_counter[d] += 1
    
    def total_conflicts_from_counter(counter):
        return sum(v - 1 for v in counter.values() if v > 1)
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Choose a row to reconstruct
        r = rng.randint(0, n_rows - 1)
        
        # Remove row r's diffs from global counter
        old_rd = row_diffs_list[r]
        for d in old_rd:
            global_diff_counter[d] -= 1
            if global_diff_counter[d] == 0:
                del global_diff_counter[d]
        
        old_scope = max(current_rows[r])
        old_conflicts_row = sum(1 for d in old_rd if global_diff_counter.get(d, 0) > 0)
        
        # Generate new row
        # Try to keep scope similar or smaller
        max_val = max(current_scope, old_scope)
        new_row = random_row(max_val)
        
        if new_row is None:
            # Restore
            for d in old_rd:
                global_diff_counter[d] = global_diff_counter.get(d, 0) + 1
            continue
        
        new_rd = get_row_diffs(new_row)
        new_scope = max(new_row)
        
        # Count conflicts of new row with others
        new_conflicts_row = sum(1 for d in new_rd if global_diff_counter.get(d, 0) > 1)
        
        # Compute delta in total conflicts
        # Old contribution: conflicts involving row r
        # This is tricky; let's just recompute total for simplicity in early steps,
        # but for speed, approximate:
        
        # Actually, let's just compute the change:
        # When we remove old row, conflicts decrease by: for each d in old_rd, if counter[d] was >1, 
        # removing it reduces conflicts by 1 for each other copy.
        # This is complex. Let's just do a full recount every N steps, or use a simpler metric.
        
        # Simpler: use a proxy objective = sum of (max entry) + lambda * (number of diffs shared with others)
        # But we want validity. Let's use:
        # objective = (new_scope if new_scope > current_scope else current_scope) + 100 * new_conflicts_row
        
        # For SA, we need a consistent objective. Let's use:
        # E = total_conflicts * 1000 + max_scope
        # But recomputing total_conflicts is expensive.
        
        # Alternative: use a local objective for the row move:
        # delta_E = (new_conflicts_row - old_conflicts_row) * 1000 + (new_scope - old_scope)
        # This isn't exactly the global change, but it's a reasonable heuristic for SA.
        
        delta_conflicts = new_conflicts_row - old_conflicts_row
        delta_scope = new_scope - old_scope
        delta_E = delta_conflicts * 1000 + delta_scope
        
        accept = False
        if delta_E <= 0:
            accept = True
        else:
            if T > 0.01:
                import math
                if rng.random() < math.exp(-delta_E / T):
                    accept = True
        
        if accept:
            current_rows[r] = new_row
            row_diffs_list[r] = new_rd
            for d in new_rd:
                global_diff_counter[d] = global_diff_counter.get(d, 0) + 1
            
            # Update best
            # Recompute actual conflicts periodically
            if step % 50 == 0 or time.time() > deadline - 10:
                actual_conflicts = total_conflicts_from_counter(global_diff_counter)
                actual_scope = max(max(row) for row in current_rows)
                if actual_conflicts == 0 and actual_scope < best_scope:
                    best_scope = actual_scope
                    best_rows = [list(row) for row in current_rows]
                    best_conflicts = 0
                    return best_rows
                elif actual_conflicts < best_conflicts:
                    best_conflicts = actual_conflicts
                    best_rows = [list(row) for row in current_rows]
                    best_scope = actual_scope
        else:
            # Restore old row
            for d in old_rd:
                global_diff_counter[d] = global_diff_counter.get(d, 0) + 1
        
        # Cool down
        T *= cooling
        
        # If we've found a valid solution, try to improve scope with a focused search
        if best_conflicts == 0:
            # Do a local search to reduce scope
            # Try replacing rows with smaller max values
            for _ in range(1000):
                if time.time() > deadline - 5:
                    break
                r = rng.randint(0, n_rows - 1)
                old_row = current_rows[r]
                old_max = max(old_row)
                # Try to find a row with smaller max
                target_max = old_max - rng.randint(1, 5)
                if target_max < k:
                    continue
                new_row = random_row(target_max)
                if new_row is None:
                    continue
                new_rd = get_row_diffs(new_row)
                # Check if it conflicts with others
                old_rd = row_diffs_list[r]
                for d in old_rd:
                    global_diff_counter[d] -= 1
                    if global_diff_counter[d] == 0:
                        del global_diff_counter[d]
                new_conflicts = sum(1 for d in new_rd if global_diff_counter.get(d, 0) > 0)
                # Restore
                for d in old_rd:
                    global_diff_counter[d] = global_diff_counter.get(d, 0) + 1
                if new_conflicts == 0:
                    current_rows[r] = new_row
                    row_diffs_list[r] = new_rd
                    for d in new_rd:
                        global_diff_counter[d] = global_diff_counter.get(d, 0) + 1
                    for d in old_rd:
                        global_diff_counter[d] -= 1
                        if global_diff_counter[d] == 0:
                            del global_diff_counter[d]
                    # Update best
                    actual_scope = max(max(row) for row in current_rows)
                    if actual_scope < best_scope:
                        best_scope = actual_scope
                        best_rows = [list(row) for row in current_rows]
                        return best_rows
    
    # Return best found
    return best_rows
