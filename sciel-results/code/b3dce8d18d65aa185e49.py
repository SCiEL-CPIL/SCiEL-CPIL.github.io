import time
import random

def construct(n_rows, k):
    # n_rows=4, k=7 (8 numbers)
    # We want to minimize M (max entry)
    
    start_time = time.time()
    # Safety margin for the 150s limit. Let's aim for 135s of work.
    deadline = start_time + 135.0

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        """All positive differences in a row (as a list)."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def row_diffs_set(row):
        """All positive differences in a row (as a set)."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
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
    def try_reduce_row(rows, row_idx, used_other, max_attempts=150):
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
            # Jitter increases with attempts to explore more
            jitter = 0.15 if attempt > 15 else 0.0
            while len(row) < k + 1:
                found = False
                # Look ahead a bit
                # We need to find a valid x. If we jitter, we might skip valid ones.
                # We should ensure we don't skip forever.
                # The range x+1 to x+80 is a heuristic window.
                # If no candidate is found in the window, we might fail this attempt.
                for trial_x in range(x + 1, x + 80):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        # Apply jitter: sometimes skip a valid candidate to explore
                        if jitter > 0 and random.random() < jitter and trial_x < 250:
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
                # Verify the whole set is valid with this new row
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
        max_passes = 6
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            # Sort indices by descending max entry to target the largest rows first
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=80)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    def kick(rows):
        """Perturb the solution to escape local minima.
        
        Uses conflict-aware candidate selection: among valid candidates, prefer
        smaller values (tighter packing) but with randomness to explore. When no
        valid candidate exists in the window, fall back to the least-conflicting
        candidate to maintain forward progress.
        """
        rows = [list(r) for r in rows]
        # Strategy: Rebuild 1 or 2 random rows with high randomness
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        
        for idx in indices:
            # Calculate used diffs from other rows
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            
            # Rebuild the row with conflict-aware selection
            success = False
            for attempt in range(30):
                row, x = [0], 0
                current_row_diffs = set()
                valid_row = True
                # Jitter probability decreases with attempt to converge
                jitter_prob = max(0.15, 0.5 - attempt * 0.012)
                while len(row) < k + 1:
                    found = False
                    best_valid_x = None
                    best_conflicts = float('inf')
                    best_conflict_x = None
                    best_conflict_new = None
                    # Wider search range for kicks
                    window = 100
                    for trial_x in range(x + 1, x + window):
                        new = {trial_x - a for a in row}
                        conflicts = len(new & used_other) + len(new & current_row_diffs)
                        if conflicts == 0:
                            # Valid candidate: prefer smaller, but jitter to explore
                            if best_valid_x is None or trial_x < best_valid_x:
                                best_valid_x = trial_x
                                best_conflicts = 0
                            # Apply jitter: sometimes skip this valid candidate
                            if random.random() < jitter_prob and trial_x < 400:
                                continue
                            row.append(trial_x)
                            current_row_diffs |= new
                            found = True
                            break
                        else:
                            if conflicts < best_conflicts:
                                best_conflicts = conflicts
                                best_conflict_x = trial_x
                                best_conflict_new = new
                    if not found:
                        # Fallback: use least-conflicting candidate if conflicts are low
                        if best_conflict_x is not None and best_conflicts < 8:
                            row.append(best_conflict_x)
                            current_row_diffs |= best_conflict_new
                        else:
                            valid_row = False
                            break
                    x = row[-1]
                
                if valid_row:
                    rows[idx] = row
                    success = True
                    break
            
            if not success:
                # If we failed to rebuild, use a greedy row as fallback
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
    consecutive_no_improve = 0
    max_no_improve = 25
    while time_left() > 20 and consecutive_no_improve < max_no_improve:
        # 1. Local Search
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
            consecutive_no_improve = 0
        
        # 2. Kick
        if time_left() < 15:
            break
            
        kicked_rows = kick(best_rows)
        
        # 3. Local Search on kicked rows
        polished_kicked = local_polish(kicked_rows)
        
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    # Final high-precision polish pass: tighter search windows, more attempts
    if time_left() > 8:
        def final_polish(rows):
            """Tighter local optimization with more attempts per row."""
            rows = [list(r) for r in rows]
            improved = True
            max_passes = 10
            pass_count = 0
            while improved and time_left() > 4 and pass_count < max_passes:
                improved = False
                pass_count += 1
                order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
                for i in order:
                    if time_left() < 4:
                        break
                    used_other = set()
                    for j in range(n_rows):
                        if j != i:
                            used_other |= row_diffs_set(rows[j])
                    new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=200)
                    if new_max < max(rows[i]):
                        old_scope = scope(rows)
                        rows[i] = new_row
                        new_scope = scope(rows)
                        if new_scope < old_scope:
                            improved = True
            return rows

        polished = final_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    return best_rows
