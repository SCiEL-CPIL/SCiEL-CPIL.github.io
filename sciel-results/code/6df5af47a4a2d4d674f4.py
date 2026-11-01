"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        """All positive differences in a row (as a list)."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_list(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline (fast, deterministic) ----------
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)

    # ---------- Per-row local optimization (refinement machinery) ----------
    def row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def try_reduce_row(rows, row_idx, used_other, max_attempts=100):
        """Try to find a row at row_idx with smaller max entry (same k+1 size)."""
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 2:
                break
            row, x = [0], 0
            current_row_diffs = set()
            success = True
            # Randomized greedy for this row to escape local minima in row construction
            jitter = 0.1 if attempt > 10 else 0.0
            while len(row) < k + 1:
                found = False
                # Look ahead a bit
                for trial_x in range(x + 1, x + 50):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        # Apply jitter: sometimes skip a valid candidate to explore
                        if jitter > 0 and random.random() < jitter and trial_x < 200:
                            continue
                        row.append(trial_x)
                        current_row_diffs |= new
                        found = True
                        break
                if not found:
                    success = False
                    break
                x = row[-1]
            if success and max(row) < best_max:
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = row
                if is_valid(test_rows):
                    best_row = list(row)
                    best_max = max(row)
        return best_row, best_max

    def local_polish(rows):
        """Reduce scope by locally re-optimizing each row (ordered by scope)."""
        rows = [list(r) for r in rows]
        improved = True
        # Limit the number of full passes to avoid spending too much time here
        # if we are in an ILS loop
        max_passes = 5 
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=50)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    def kick(rows):
        """Perturb the solution using conflict-directed value removal (CDVR)."""
        rows = [list(r) for r in rows]
        
        # 1. Compute global conflict counts for each difference
        diff_counts = {}
        for r_idx, row in enumerate(rows):
            diffs = row_diffs_set(row)
            for d in diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
        
        # 2. Identify the most conflicting difference
        # If no conflicts, the solution is valid. We can still kick to explore.
        # If conflicts exist, find the diff with the highest count.
        max_conflict = 0
        worst_diff = None
        for d, c in diff_counts.items():
            if c > max_conflict:
                max_conflict = c
                worst_diff = d
        
        # 3. Select rows to modify
        # We want to modify rows that contribute to the worst conflict, or random rows if no conflict.
        if worst_diff is not None:
            # Find rows containing worst_diff
            conflict_rows = []
            for r_idx, row in enumerate(rows):
                if worst_diff in row_diffs_set(row):
                    conflict_rows.append(r_idx)
            
            # Pick 1 or 2 random rows from the conflict set
            num_kicks = min(len(conflict_rows), random.choice([1, 2]))
            indices = random.sample(conflict_rows, num_kicks)
        else:
            # No conflicts, pick random rows to explore neighborhood
            num_kicks = random.choice([1, 2])
            indices = random.sample(range(n_rows), num_kicks)
        
        # 4. Rebuild selected rows with guided randomness
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            
            # Try to rebuild the row, prioritizing values that don't create new conflicts
            # We use a "soft" constraint: prefer values that avoid the most frequent diffs in used_other
            # But we must strictly avoid any diff in used_other.
            
            best_new_row = None
            best_new_max = float('inf')
            
            # Try multiple random constructions
            for attempt in range(30):
                row, x = [0], 0
                current_row_diffs = set()
                valid_row = True
                
                while len(row) < k + 1:
                    found = False
                    # Look ahead
                    # Sort candidates by "safety": how many new diffs they create that are close to used_other?
                    # Simple heuristic: just randomize the order of valid candidates slightly
                    candidates = []
                    for trial_x in range(x + 1, x + 150):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            candidates.append(trial_x)
                    
                    if not candidates:
                        valid_row = False
                        break
                    
                    # Pick a candidate. 
                    # If we are trying to fix a conflict, maybe pick the smallest? 
                    # Or random? 
                    # Let's mix: 50% smallest, 50% random from first 20 valid.
                    if random.random() < 0.5 and len(candidates) > 1:
                        # Pick from the smaller end to keep scope low
                        pick_range = min(10, len(candidates))
                        choice = random.choice(candidates[:pick_range])
                    else:
                        choice = candidates[0]
                    
                    row.append(choice)
                    current_row_diffs |= {choice - a for a in row[:-1]}
                    x = choice
                
                if valid_row:
                    new_max = max(row)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_new_row = row
            
            if best_new_row:
                rows[idx] = best_new_row
            else:
                # Fallback to greedy
                greedy_row, x = [0], 0
                row_diffs = set()
                while len(greedy_row) < k + 1:
                    x += 1
                    new = {x - a for a in greedy_row}
                    if not (new & used_other) and not (new & row_diffs):
                        greedy_row.append(x)
                        row_diffs |= new
                rows[idx] = greedy_row

        return rows

    # Initial polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Iterated Local Search
    # We use a loop that alternates between Local Search and Kicks
    # This is structurally different from a single long SA run.
    # Add a "restart" mechanism if stuck
    stuck_count = 0
    max_stuck = 5
    
    while time_left() > 20:
        # 1. Local Search
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
            stuck_count = 0
        else:
            stuck_count += 1
        
        # 2. Kick
        if time_left() < 15:
            break
            
        # If stuck for a while, use a stronger kick (random restart of a row)
        if stuck_count >= max_stuck:
            # Force a random rebuild of a random row to escape the basin
            idx = random.randint(0, n_rows - 1)
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(best_rows[j])
            
            # Greedy rebuild with high jitter
            row, x = [0], 0
            current_row_diffs = set()
            valid_row = True
            while len(row) < k + 1:
                found = False
                for trial_x in range(x + 1, x + 200):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        if random.random() < 0.3:
                            continue
                        row.append(trial_x)
                        current_row_diffs |= new
                        found = True
                        break
                if not found:
                    valid_row = False
                    break
                x = row[-1]
            
            if valid_row:
                new_rows = [list(r) for r in best_rows]
                new_rows[idx] = row
                best_rows = new_rows
                stuck_count = 0
            else:
                # If even that fails, we are in trouble. Break or try another idx.
                # Let's just break to avoid infinite loop of failures.
                break

        kicked_rows = kick(best_rows)
        
        # 3. Local Search on kicked rows
        polished_kicked = local_polish(kicked_rows)
        
        # 4. Update best
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)
        else:
            # If the kick didn't lead to improvement, we might stick to the previous best
            # and try a different kick next iteration.
            # This is the standard ILS behavior.
            pass

    # Final polish to ensure we squeeze out any last drops
    if time_left() > 5:
        # Run multiple passes of local polish in the remaining time
        while time_left() > 2:
            polished = local_polish(best_rows)
            if scope(polished) < best_scope:
                best_rows = polished
                best_scope = scope(polished)
            else:
                break

    # Final validity guarantee.
    if not is_valid(best_rows):
        # Fallback to a fresh greedy (always valid).
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
